/*
 * Copyright (c) 2011-2020, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.sfm.structure2;

import boofcv.alg.sfm.structure2.PairwiseImageGraph2.Motion;
import boofcv.alg.sfm.structure2.SceneWorkingGraph.View;
import lombok.Getter;
import org.ddogleg.sorting.QuickSort_F64;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.GrowQueue_F64;
import org.ddogleg.struct.VerbosePrint;
import org.jetbrains.annotations.Nullable;

import java.io.PrintStream;
import java.util.*;

import static boofcv.misc.BoofMiscOps.assertBoof;

/**
 * Selects a subset of views from a {@link SceneWorkingGraph} as the first step before performing local bundle
 * adjustment. The goal is to select the subset of views which would contribute the most to the target view's
 * state estimate. To keep computational limits in check the user needs to specifies a maximum number of views.
 *
 * Every connection between views is assigned a score. The goal is to find the subset of views which maximizes
 * the minimum score across all connections between views. This is approximated using the greedy algorithm below:
 *
 * Summary:
 * <ol>
 *     <li>Set of views will be the target's neighbors and their neighbors</li>
 *     <li>Create a list of all edges and their scores that connect these</li>
 *     <li>Repeat the loop below until only N views remain</li>
 *     <ol>
 *     <li>Select the edge with the lowest score</li>
 *     <li>Pick one of the views it is connected to be prune based on their connections</li>
 *     <li>Remove the view and all the edges connected to it</li>
 *     </ol>
 * </ol>
 *
 * Care is taken so that not all the direct neighbors are removed by setting a {@link #minNeighbors minimum number}
 * that must remain at the end. If all neighbors are removed then there is the possibility that all the remaining
 * nodes will not see the same features as the target.
 *
 * <p>
 * WARNING: There is a known flaw where a multiple disconnected graphs can be created in a poorly connected graph<br>
 * </p>
 *
 * @author Peter Abeles
 */
public class SelectNeighborsAroundView implements VerbosePrint {

	/** Moving number of views in the constructed working graph. */
	public int maxViews = 10;

	/** When a view is scored for removal the N-worstOfTop connection is used for the score. */
	public int worstOfTop = 3;

	/** There should be at least this number of direct neighbors in the final list */
	public int minNeighbors = 2;

	/** Score function used to evaluate which motions should be used */
	public PairwiseGraphUtils.ScoreMotion scoreMotion = new PairwiseGraphUtils.DefaultScoreMotion();

	/** Copy of the local scene which can be independently optimized */
	protected @Getter final SceneWorkingGraph localWorking = new SceneWorkingGraph();

	private @Nullable PrintStream verbose;

	//-------------- Internal Working Space

	// Keeps track of how many direct connections to the target remain
	int remainingNeighbors;

	// List of all views that it will consider for inclusion in the sub-graph
	List<View> candidates = new ArrayList<>();
	// Fast look up of candidates
	Map<String, View> lookup = new HashMap<>();
	// List of all edges with scores
	FastQueue<EdgeScore> edges = new FastQueue<>(EdgeScore::new);
	// indicates if a view has been included in an inlier set
	Set<String> hasFeatures = new HashSet<>();

	// storage for the score of connected edges
	GrowQueue_F64 connScores = new GrowQueue_F64();
	QuickSort_F64 sorter = new QuickSort_F64();

	/**
	 * Computes a local graph with the view as a seed. Local graph can be accessed by calling {@link #getLocalWorking()}
	 *
	 * @param target (Input) The view that a local graph is to be built around
	 * @param working (Input) A graph of the entire scene that a sub graph is to be made from
	 */
	public void process( View target, SceneWorkingGraph working ) {
		initialize();

		// Create the initial list of candidate views for inclusion in the sub graph
		addNeighbors2(target, working);
		// Prune the number of views down to the minimal number
		pruneViews(target);

		// Create the local working graph that can be optimized
		createLocalGraph(target, working);
	}

	void initialize() {
		edges.reset();
		candidates.clear();
		lookup.clear();
		localWorking.reset();
	}

	/**
	 * Adds target's neighbors and their neighbors
	 */
	void addNeighbors2( View seed, SceneWorkingGraph working ) {
		// put the target into the lookup list to avoid double counting. Target isn't a candidate since it's mandatory
		lookup.put(seed.pview.id, seed);

		// Add immediate neighbors to candidate list
		for (int connIdx = 0; connIdx < seed.pview.connections.size; connIdx++) {
			Motion m = seed.pview.connections.get(connIdx);
			View o = working.lookupView(m.other(seed.pview).id);
			// No corresponding working view
			if( o == null )
				continue;

			addCandidateView(o);
		}

		// Note the number of neighbors since we will keep track of how many are left in the candidates list
		remainingNeighbors = candidates.size();

		// Add the neighbor's neighbors to the candidate list
		for (int candIdx = candidates.size() - 1; candIdx >= 0; candIdx--) {
			View c = candidates.get(candIdx);
			for (int connIdx = 0; connIdx < c.pview.connections.size; connIdx++) {
				Motion m = c.pview.connections.get(connIdx);
				View o = working.lookupView(m.other(c.pview).id);
				// No corresponding working view
				if( o == null )
					continue;
				// don't add if it's already in there
				if (lookup.containsKey(o.pview.id))
					continue;
				addCandidateView(o);
			}
		}

		// Now that all the node have been we can create the edges
		addEdgesOf(working, seed);
		for (int candIdx = 0; candIdx < candidates.size(); candIdx++) {
			View c = candidates.get(candIdx);
			addEdgesOf(working, c);
		}
	}

	void addCandidateView( View o ) {
		candidates.add(o);
		lookup.put(o.pview.id, o);
	}

	/**
	 * Add all the edges/connections for view 'c' that point to a candidate
	 */
	private void addEdgesOf( SceneWorkingGraph working, View c ) {
		for (int connIdx = 0; connIdx < c.pview.connections.size; connIdx++) {
			Motion m = c.pview.connections.get(connIdx);
			View o = working.lookupView(m.other(c.pview).id);
			// No corresponding working view
			if( o == null )
				continue;
			// If it's not pointing too a candidate or the target, do not add this edge
			if (!lookup.containsKey(o.pview.id))
				continue;
			// To avoid adding the same edge twice use the unique ID
			if (c.pview.id.compareTo(o.pview.id) >= 0)
				continue;
			addEdge(m);
		}
	}

	/**
	 * Reduces the candidate size until the requested maximum number of views has been meet
	 */
	void pruneViews( View seed ) {
		if (verbose != null) verbose.println("ENTER pruneViews target=" + seed.pview.id);
		// maxViews-1 because the target is not in the candidates list
		while (candidates.size() > maxViews - 1) {

			// Search for the edge with the lowest score. The reason we don't just sort once and be done with it
			// is that each time we remove an element from the list that's an O(N) operation or remove swap O(1)
			// but need to re-sort it
			int lowestIndex = -1;
			double lowestScore = Double.MAX_VALUE;
			for (int i = 0; i < edges.size; i++) {
				EdgeScore s = edges.get(i);
				if (s.score >= lowestScore)
					continue;
				// See if too many neighbors have been removed and that it should keep the remaining
				if (remainingNeighbors <= minNeighbors && s.m.isConnected(seed.pview))
					continue;
				// Check here if removing the node would cause a split. If so don't mark it as the lowest
				lowestScore = s.score;
				lowestIndex = i;
			}

			if (lowestIndex < 0)
				throw new RuntimeException("Highly likely that this is miss configured. No valid candidate has been " +
						"found for removal. Is 'minNeighbors' >= maxViews-1?");

			if (verbose != null) verbose.println("Pruning score=" + lowestScore + " " + edges.get(lowestIndex).m);

			Motion m = edges.get(lowestIndex).m;

			if (m.isConnected(seed.pview)) {
				// Remove a neighbor of the target. No need to select which one to remove since the
				// target can't be removed
				removeCandidateNode(m.other(seed.pview).id, seed);
				if (verbose != null)
					verbose.println("Connects to target. No need to score. neighbors=" + remainingNeighbors);
			} else {
				// be careful to not remove a view which links to the seed if it's below the limit
				double scoreSrc = (remainingNeighbors <= minNeighbors && m.src.findMotion(seed.pview) != null) ?
						Double.MAX_VALUE : scoreForRemoval(m.src, m);
				double scoreDst = (remainingNeighbors <= minNeighbors && m.dst.findMotion(seed.pview) != null) ?
						Double.MAX_VALUE : scoreForRemoval(m.dst, m);
				if (verbose != null) verbose.println("Scores: src=" + scoreSrc + " dst=" + scoreDst);
				removeCandidateNode(((scoreSrc < scoreDst) ? m.src : m.dst).id, seed);
			}

			// WARNING: This should be made a bit less naive by considering if removing a View would cause other
			// Views to have no path to the target. This could probably be done efficiently by saving a reference
			// towards the target view

			// WARNING: There is nothing stopping it from pruning all of the target's neighbors also!
		}
	}

	/**
	 * Score the quality of a View based on the worst score of its top N connections.
	 *
	 * @param v The view being considered for removal
	 * @param ignore skips this motion
	 * @return score
	 */
	double scoreForRemoval( PairwiseImageGraph2.View v, Motion ignore ) {
		connScores.reset();

		for (int connIdx = 0; connIdx < v.connections.size; connIdx++) {
			Motion m = v.connections.get(connIdx);
			if (m == ignore)
				continue;
			String o = m.other(v).id;
			if (!lookup.containsKey(o)) {
				continue;
			}
			connScores.add(scoreMotion.score(m));
		}
		if (connScores.size == 0)
			return 0.0;

		// Find the N-worstOfTop best hypothesis
		connScores.sort(sorter);
		int idx = Math.max(0, connScores.size - worstOfTop);
		return connScores.get(idx);
	}

	/**
	 * Removes the specified view from the candidate list and then searches for all of its
	 * edges in the edge list and removes those
	 *
	 * @param id View id that is to be removed from candidate list
	 */
	void removeCandidateNode( String id, View seed ) {
		if (verbose != null) verbose.println("Removing candidate view='" + id + "'");

		// Remove the specified node from the candidate list data structures
		View v = lookup.remove(id);
		assertBoof(candidates.remove(v));

		// Keep track of how many direct neighbors to the target have been removed
		if (null != v.pview.findMotion(seed.pview)) {
			remainingNeighbors--;
			if (verbose != null)
				verbose.println("Neighbor of seed has been removed. view='" + id + "'  remaining=" + remainingNeighbors);
		}

		// Remove all edges that point to this view in the edge list
		for (int connIdx = 0; connIdx < v.pview.connections.size; connIdx++) {
			Motion m = v.pview.connections.get(connIdx);
			View o = lookup.get(m.other(v.pview).id);

			// If by removing the target one of it's connections is now an orphan remove that note
			// TODO see comment about graph splits. This should be handled by that logic and this removed
			if (isOrphan(o)) {
				if (verbose != null) verbose.println("Removing orphaned view='" + o.pview.id + "'");
				assertBoof(null != lookup.remove(o.pview.id),"Not in lookup list");
				assertBoof(candidates.remove(o), "Can't remove. Not in candidate list");
			}

			boolean found = false;
			for (int i = 0; i < edges.size; i++) {
				if (edges.get(i).m != m)
					continue;
				edges.removeSwap(i);
				found = true;
				break;
			}
			assertBoof(found, "No matching edge found. BUG. id='" + id + "' m.other='" + o.pview.id + "'");
		}
	}

	/**
	 * Checks to see if the View has any connections to another candidate
	 */
	boolean isOrphan( View v ) {
		for (int connIdx = 0; connIdx < v.pview.connections.size; connIdx++) {
			Motion m = v.pview.connections.get(connIdx);
			String o = m.other(v.pview).id;
			if (lookup.containsKey(o))
				return false;
		}
		return true;
	}

	/**
	 * Create a local graph from all the candidate Views
	 *
	 * @param seed Target view from original graph that the local graph is made around
	 * @param working The original graph
	 */
	void createLocalGraph( View seed, SceneWorkingGraph working ) {
		hasFeatures.clear();
		addViewToLocal(seed);
		for (int i = 0; i < candidates.size(); i++) {
			addViewToLocal(candidates.get(i));
		}
		// For views with no features look at its neighbors to see if it's an inlier then add those
		for (int i = 0; i < candidates.size(); i++) {
			View origView = localWorking.lookupView(candidates.get(i).pview.id);
			Objects.requireNonNull(origView);

			if (hasFeatures.contains(origView.pview.id))
				continue;

			// Search for a connected view which has this view inside its inlier list. There has to be
			// at least one
			for (int connIdx = 0; connIdx < origView.pview.connections.size; connIdx++) {
				Motion m = origView.pview.connections.get(connIdx);
				View v = working.lookupView(m.other(origView.pview).id);
				if( v == null )
					continue;

				if (v.inliers.isEmpty())
					continue;
				for (int j = 0; j < v.inliers.views.size; j++) {
					if (!v.inliers.views.get(j).id.equals(origView.pview.id))
						continue;

					// Create an inlier list
					origView.inliers.views.add(origView.pview);
					origView.inliers.observations.grow().setTo(v.inliers.observations.get(j));

					// Use the first inlier list that we found
					break;
				}

				if (!origView.inliers.isEmpty())
					break;
			}
			assertBoof(!origView.inliers.isEmpty(), "BUG! there can be no estimated state if it was never in an " +
					"inlier list of a neighbor. view.id=" + origView.pview.id);
		}
	}

	/**
	 * Adds the view in the original graph to the new local one while copying over all the useful information
	 *
	 * @param origView View in the original graph that was passed in
	 */
	void addViewToLocal( View origView ) {
		// Create the local node
		View localView = localWorking.addView(origView.pview);

		// copy geometric information over
		localView.intrinsic.set(origView.intrinsic);
		localView.world_to_view.set(origView.world_to_view);
		localView.imageDimension.setTo(origView.imageDimension);

		// If this isn't a view with inliers move on
		if (origView.inliers.isEmpty())
			return;

		// Copy over inliers that are in the sub graph
		for (int i = 0; i < origView.inliers.views.size; i++) {
			PairwiseImageGraph2.View pview = origView.inliers.views.get(i);
			// see if it's in the sub graph
			if (!lookup.containsKey(pview.id))
				continue;
			// mark this as having features assigned to it
			hasFeatures.add(pview.id);

			// Add the inliers to the local view
			localView.inliers.views.add(pview);
			localView.inliers.observations.grow().setTo(origView.inliers.observations.get(i));
		}
	}

	private void addEdge( Motion m ) {
		EdgeScore edgeScore = edges.grow();
		edgeScore.m = m;
		edgeScore.score = scoreMotion.score(m);
	}

	@Override
	public void setVerbose( @Nullable PrintStream out, @Nullable Set<String> configuration ) {
		this.verbose = out;
	}

	static class EdgeScore {
		// The motion this edge references
		Motion m;
		// quality of geometric information in this edge. Higher is better
		double score;
	}
}
