/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.scene.nister2006;

import boofcv.alg.scene.vocabtree.HierarchicalVocabularyTree;
import boofcv.alg.scene.vocabtree.HierarchicalVocabularyTree.Node;
import boofcv.misc.BoofLambdas;
import boofcv.misc.BoofMiscOps;
import gnu.trove.impl.Constants;
import gnu.trove.map.TIntFloatMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntFloatHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import lombok.Getter;
import lombok.Setter;
import org.ddogleg.struct.DogArray;
import org.ddogleg.struct.DogArray_I32;

import java.util.Collections;
import java.util.List;

/**
 * Image recognition based off of [1] using inverted files. A {@link HierarchicalVocabularyTree} is assumed to hav
 * e been already trained. When an image is added to the database a TF-IDF descriptor is computed using the tree
 * and then added to the relevant tree's leaves. When an image is looked up its TF-IDF descriptor is found then
 * all images in the data base are found that share at least one leaf node. These candidate matches are then
 * compared against each other and scored using L2-Norm.
 *
 * <p>
 * [1] Nister, David, and Henrik Stewenius. "Scalable recognition with a vocabulary tree."
 * 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. Ieee, 2006.
 * </p>
 *
 * @author Peter Abeles
 */
public class RecognitionVocabularyTreeNister2006<Point> {

	/** Vocabulary Tree */
	public @Getter @Setter HierarchicalVocabularyTree<Point, LeafData> tree;

	/** List of images added to the database */
	protected @Getter final DogArray<ImageInfo> imagesDB = new DogArray<>(ImageInfo::new, ImageInfo::reset);
	/** Scores for all candidate images which have been sorted */
	protected @Getter final DogArray<Match> matchScores = new DogArray<>(Match::new, Match::reset);

	//---------------- Internal Workspace

	// The "frequency" that nodes in the tree appear in this image
	protected final DogArray<Frequency> frequencies = new DogArray<>(Frequency::new, Frequency::reset);
	protected final TIntObjectMap<Frequency> nodeFrequencies = new TIntObjectHashMap<>();

	// Temporary storage for an image's TF description while it's being looked up
	protected final TIntFloatMap tempDescTermFreq = new TIntFloatHashMap();

	// Predeclare array for storing keys. Avoids unnecessary array creation
	protected final DogArray_I32 keys = new DogArray_I32();

	/**
	 * Adds a new image to the database.
	 *
	 * @param imageID The image's unique ID for later reference
	 * @param imageFeatures Feature descriptors from an image
	 * @param cookie Optional user defined data which will be attached to the image
	 */
	public void add( int imageID, List<Point> imageFeatures, Object cookie ) {
		ImageInfo info = imagesDB.grow();
		info.imageId = imageID;
		info.cookie = cookie;

		// compute a descriptor for this image while adding it to the leaves
		describe(imageFeatures, info.descTermFreq, ( leafNode ) -> {
			// Add the image info to the leaf if it hasn't already been added
			LeafData leafData = tree.listData.get(leafNode.dataIdx);
			if (!leafData.images.containsKey(info.imageId)) {
				leafData.images.put(info.imageId, info);
			}
		});
	}

	/**
	 * Looks up the best match from the data base. The list of all potential matches can be accessed by calling
	 * {@link #getMatchScores()}.
	 *
	 * @param imageFeatures Feature descriptors from an image
	 * @return The best matching image with score from the database
	 */
	public Match lookup( List<Point> imageFeatures ) {
		TIntSet candidates = new TIntHashSet();
		matchScores.reset();

		// Create a description of this image and collect potential matches from leaves
		describe(imageFeatures, tempDescTermFreq, ( leafNode ) -> {
			LeafData leafData = tree.listData.get(leafNode.dataIdx);
			for (int i = 0; i < leafData.images.size(); i++) {
				ImageInfo c = leafData.images.get(i);
				if (!candidates.add(c.imageId))
					continue;
				matchScores.grow().image = c;
			}
		});

		for (int i = 0; i < matchScores.size(); i++) {
			Match m = matchScores.get(i);
			m.score = distanceL2Norm(tempDescTermFreq, m.image.descTermFreq);
		}

		Collections.sort(matchScores.toList());

		return matchScores.get(0);
	}

	protected void describe( List<Point> imageFeatures, TIntFloatMap descTermFreq, BoofLambdas.ProcessObject<Node> op ) {
		// Reset work variables
		frequencies.reset();
		nodeFrequencies.clear();
		descTermFreq.clear();

		// Sum of the weight of all graph nodes it sees
		for (int descIdx = 0; descIdx < imageFeatures.size(); descIdx++) {
			int leafNodeIdx = tree.searchPathToLeaf(imageFeatures.get(descIdx), ( node ) -> {
				Frequency f = nodeFrequencies.get(node.id);
				if (f == null) {
					f = frequencies.grow();
					f.node = node;
					nodeFrequencies.put(node.id, f);
				}
				f.sum += node.weight;
			});

			// Process the leaf node in the passed in operation
			op.process(tree.nodes.get(leafNodeIdx));
		}

		//------ Create the TF-IDF descriptor for this image
		// Compute the sum to ensure the F2-norm of the descriptor is 1
		double sum = 0.0;
		for (int i = 0; i < nodeFrequencies.size(); i++) {
			sum += nodeFrequencies.get(i).sum;
		}
		BoofMiscOps.checkTrue(sum != 0.0, "Sum of weights is zero. Something went very wrong");

		for (int i = 0; i < nodeFrequencies.size(); i++) {
			Frequency f = nodeFrequencies.get(i);
			descTermFreq.put(f.node.id, (float)(f.sum/sum));
		}
	}

	/**
	 * Computes L2-Norm for score between the two descriptions. Searches for common non-zero elements between
	 * the two then uses the simplified equation from [1].
	 */
	public float distanceL2Norm( TIntFloatMap descA, TIntFloatMap descB ) {
		// Get the key and make sure it doesn't declare new memory
		keys.resize(descA.size());
		descA.keys(keys.data);

		// Compute dot product of common non-zero elements
		float sum = 0.0f;
		for (int keyIdx = 0; keyIdx < keys.size; keyIdx++) {
			int key = keys.data[keyIdx];

			float valueA = descA.get(key);
			float valueB = descB.get(key);
			if (valueB < 0.0f)
				continue;

			sum += valueA*valueB;
		}

		return 2.0f - 2.0f*sum;
	}

	/** Information about an image stored in the database */
	public static class ImageInfo {
		/** TF-IDF description of the image. Default -1 for no key and no value. */
		public TIntFloatMap descTermFreq = new TIntFloatHashMap(
				Constants.DEFAULT_CAPACITY, Constants.DEFAULT_LOAD_FACTOR, -1, -1);

		/** Use specified data associated with this image */
		public Object cookie;
		/** Unique ID for this image */
		public int imageId;

		public <T> T getCookie() {
			return (T)cookie;
		}

		public void reset() {
			descTermFreq.clear();
			cookie = null;
			imageId = -1;
		}
	}

	// Used to sum the frequency of words (graph nodes) in the image
	protected static class Frequency {
		// sum of weights
		double sum;
		// The node which is referenced
		Node node;

		public void reset() {
			sum = 0;
			node = null;
		}
	}

	/**
	 * Match and score information.
	 */
	public static class Match implements Comparable<Match> {
		/** Fit score */
		public float score;
		/** Reference to the image in the data base that was matched */
		public ImageInfo image;

		public void reset() {
			score = 0;
			image = null;
		}

		@Override public int compareTo( Match o ) {
			return Float.compare(o.score, score);
		}
	}

	public static class LeafData {
		TIntObjectMap<ImageInfo> images = new TIntObjectHashMap<>();
	}
}
