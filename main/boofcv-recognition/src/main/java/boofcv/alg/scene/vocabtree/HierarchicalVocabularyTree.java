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

import boofcv.misc.BoofLambdas;
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
public class HierarchicalVocabularyTree<TD extends TupleDesc<TD>, Data> {

	/** Number of children for each node */
	public int branchFactor = -1;

	/** Maximum number of levels in the tree */
	public int maximumLevels = -1;

	/** Optional data associated with any of the nodes */
	public final FastArray<Data> listData;

	// list of mean descriptors that define the discretized regions
	public final PackedArray<TD> descriptions;
	// Nodes in the hierarchical tree
	// Node[0] is the root node
	public final DogArray<Node> nodes = new DogArray<>(Node::new, Node::reset);

	public HierarchicalVocabularyTree( PackedArray<TD> descriptions, Class<Data> leafType ) {
		this.descriptions = descriptions;
		listData = new FastArray<>(leafType);
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
		// assign to ID to the index. An alternative would be to use level + branch.
		n.id = index;
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
	public Node searchToLeaf( TD point, PointDistance<TD> distanceFunction ) {
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

	public int searchPathToLeaf( TD point, PointDistance<TD> distanceFunction, BoofLambdas.ProcessObject<Node> results ) {
		return 0;
	}

	/**
	 * Traverses every node in the graph (excluding the root) in a depth first manor.
	 *
	 * @param op Every node is feed into this function
	 */
	public void traverseGraphDepthFirst( BoofLambdas.ProcessObject<Node> op ) {
		FastArray<Node> queue = new FastArray<>(Node.class, maximumLevels);
		queue.add(nodes.get(0));
		DogArray_I32 branches = new DogArray_I32(maximumLevels);
		branches.add(0);

		// NOTE: Root node is intentionally skipped since it will contain all the features and has no information

		while (!nodes.isEmpty()) {
			Node n = nodes.getTail();
			int branch = branches.getTail();

			// If there are no more children to traverse in this node go back to the parent
			if (branch >= n.childrenIndexes.size) {
				nodes.removeTail();
				branches.removeTail();
				continue;
			}

			// next iteration will explore the next branch
			branches.set(branches.size - 1, branch + 1);

			// Pass in the child
			n = nodes.get(n.childrenIndexes.get(branch));
			op.process(n);

			// Can't dive into any children/branches if it's a leaf
			if (n.isLeaf())
				continue;

			queue.add(n);
			branches.add(0);
		}
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
		listData.reset();
		descriptions.reset();
		nodes.reset();
	}

	/**
	 * Adds data to the node
	 *
	 * @param node Node that data is added to
	 * @param data The data which is not associated with it
	 */
	public void addData( Node node, Data data ) {
		BoofMiscOps.checkTrue(node.dataIdx < 0);
		node.dataIdx = listData.size;
		listData.add(data);
	}

	/** Node in the Vocabulary tree */
	public static class Node {
		// How useful a match to this node is. Higher the weight, more unique it is typically.
		public double weight;
		// The unique ID assigned to this node
		public int id;
		// Which branch/child in the parent it is
		public int branch;
		// Index of the parent. The root node will have -1 here
		public int parent;
		// Index of data associated with this node
		public int dataIdx;
		// index of the first mean in the list of descriptions. Means in a set are consecutive.
		public int indexDescription;
		// index of the first child in the list of nodes. Children are consecutive.
		// If at the last level then this will point to an index in leaves
		public final DogArray_I32 childrenIndexes = new DogArray_I32();

		public boolean isLeaf() {
			return childrenIndexes.isEmpty();
		}

		public void reset() {
			weight = -1;
			id = -1;
			branch = -1;
			parent = -1;
			dataIdx = -1;
			indexDescription = -1;
			childrenIndexes.reset();
		}

		public void setTo( Node src ) {
			id = src.id;
			branch = src.branch;
			parent = src.parent;
			dataIdx = src.dataIdx;
			indexDescription = src.indexDescription;
			childrenIndexes.setTo(src.childrenIndexes);
		}
	}
}
