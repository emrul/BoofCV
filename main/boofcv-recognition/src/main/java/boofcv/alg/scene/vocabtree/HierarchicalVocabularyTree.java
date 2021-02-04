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

package boofcv.alg.scene.vocabtree;

import boofcv.misc.BoofMiscOps;
import boofcv.struct.feature.TupleDesc;
import boofcv.struct.kmeans.PackedArray;
import org.ddogleg.clustering.PointDistance;
import org.ddogleg.struct.DogArray;
import org.ddogleg.struct.DogArray_I32;
import org.ddogleg.struct.FastArray;

import static boofcv.misc.BoofMiscOps.checkTrue;

/**
 * A hierarchical tree which discretizes an N-Dimensional space. Each region in each node is defined by
 * the set of points which have mean[i] as the closest mean. Once the region is known at a level the child
 * node in the tree can then be looked up and the process repeated until it's at leaf.
 *
 * @author Peter Abeles
 **/
public class HierarchicalVocabularyTree<TD extends TupleDesc<TD>, Leaf> {

	/** Number of children for each node */
	public int branchFactor = -1;

	/** Maximum number of levels in the tree */
	public int maximumLevels = -1;

	/** All the leaves in the tree */
	public final FastArray<Leaf> leaves;

	// list of mean descriptors that define the discretized regions
	protected final PackedArray<TD> descriptions;
	// Nodes in the hierarchical tree
	// Node[0] is the root node
	protected final DogArray<Node> nodes = new DogArray<>(Node::new, Node::reset);

	// https://www.cse.unr.edu/~bebis/CS491Y/Papers/Nister06.pdf
	// https://sourceforge.net/projects/vocabularytree/
	// https://sourceforge.net/p/vocabularytree/code-0/HEAD/tree/trunk/py_vt/src/
	// http://webdiis.unizar.es/~dorian/index.php?p=31
	// https://github.com/epignatelli/scalable-recognition-with-a-vocabulary-tree

	public HierarchicalVocabularyTree( PackedArray<TD> descriptions, Class<Leaf> leafType ) {
		this.descriptions = descriptions;
		leaves = new FastArray<>(leafType);
	}

	/**
	 * Adds a new node to the graph and returns its index
	 *
	 * @param parentIndex Index of parent node.
	 * @param branch Which branch off the parent does it map to.
	 * @param desc The mean/description for this region
	 * @return Index of the newly added node
	 */
	public int addNode( int parentIndex, int branch, TD desc ) {
		int index = nodes.size;
		Node n = nodes.grow();
		n.branch = branch;
		n.indexDescription = descriptions.size();
		descriptions.addCopy(desc);
		Node parent = nodes.get(parentIndex);
		checkTrue(branch == parent.childrenIndexes.size, "Branch index must map to child index");
		n.parent = parentIndex;
		parent.childrenIndexes.add(index);
		return index;
	}

	/**
	 * Searches for the leaf node that this point belongs to
	 *
	 * @param point (Input) Point used in the search
	 * @param distanceFunction (Input) Function used to determine distance between nodes
	 * @return The found leaf node
	 */
	public Node lookupLeafNode( TD point, PointDistance<TD> distanceFunction ) {
		int level = 0;
		Node parent = nodes.get(0);

		while (level < maximumLevels) {
			if (parent.childrenIndexes.isEmpty()) {
				return parent;
			}

			int bestNodeIdx = -1;
			double bestDistance = Double.MAX_VALUE;

			for (int childIdx = 0; childIdx < parent.childrenIndexes.size; childIdx++) {
				int nodeIdx = parent.childrenIndexes.get(childIdx);

				TD desc = descriptions.getTemp(nodes.get(nodeIdx).indexDescription);
				double distance = distanceFunction.distance(point, desc);
				if (distance > bestDistance)
					continue;

				bestNodeIdx = nodeIdx;
				bestDistance = distance;
			}

			parent = nodes.get(bestNodeIdx);
			level++;
		}

		throw new RuntimeException("Invalid tree. Max depth exceeded searching for leaf");
	}

	/**
	 * Ensures it has a valid configuration
	 */
	public void checkConfig() {
		BoofMiscOps.checkTrue(branchFactor > 0, "branchFactor needs to be set");
		BoofMiscOps.checkTrue(maximumLevels > 0, "maximumLevels needs to be set");
	}

	/**
	 * Clears references to initial state but keeps allocated memory
	 */
	public void reset() {
		leaves.reset();
		descriptions.reset();
		nodes.reset();
	}

	public void setTo( HierarchicalVocabularyTree<TD, Leaf> src ) {
		branchFactor = src.branchFactor;
		maximumLevels = src.maximumLevels;

		// How to handle copy of leaves?
		// Is this function needed
	}

	/** Node in the Covabulary tree */
	public static class Node {
		// Which branch/child in the parent it is
		public int branch;
		// Index of the parent. -1 if this is at the root of the tree
		public int parent;
		// index of the first mean in the list of descriptions. Means in a set are consecutive.
		public int indexDescription;
		// index of the first child in the list of nodes. Children are consecutive.
		// If at the last level then this will point to an index in leaves
		public final DogArray_I32 childrenIndexes = new DogArray_I32();
		// Index of the leaf data it points to
		public int leaf;

		public void reset() {
			branch = -1;
			parent = -1;
			leaf = -1;
			indexDescription = -1;
			childrenIndexes.reset();
		}

		public void setTo( Node src ) {
			branch = src.branch;
			parent = src.parent;
			leaf = src.leaf;
			indexDescription = src.indexDescription;
			childrenIndexes.setTo(src.childrenIndexes);
		}
	}
}
