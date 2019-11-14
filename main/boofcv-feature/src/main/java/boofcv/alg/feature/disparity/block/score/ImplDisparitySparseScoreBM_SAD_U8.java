/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.feature.disparity.block.score;

import boofcv.alg.feature.disparity.block.DisparitySparseScoreSadRect;
import boofcv.struct.image.GrayU8;

import java.util.Arrays;

/**
 * <p>
 * Implementation of {@link DisparitySparseScoreSadRect} that processes images of type {@link GrayU8}.
 * </p>
 *
 * <p>
 * DO NOT MODIFY. Generated by {@link GenerateDisparitySparseScoreSadRect}.
 * </p>
 *
 * @author Peter Abeles
 */
public class ImplDisparitySparseScoreBM_SAD_U8 extends DisparitySparseScoreSadRect<int[],GrayU8> {

	// scores up to the maximum baseline
	int scores[];

	public ImplDisparitySparseScoreBM_SAD_U8(int minDisparity , int maxDisparity, int radiusX, int radiusY) {
		super(minDisparity,maxDisparity,radiusX, radiusY);

		scores = new int[ maxDisparity ];
	}

	@Override
	public boolean process( int x , int y ) {
		// adjust disparity for image border
		localMaxDisparity = Math.min(rangeDisparity,x-radiusX+1-minDisparity);

		if( localMaxDisparity <= 0 || x >= left.width-radiusX || y < radiusY || y >= left.height-radiusY )
			return false;

		Arrays.fill(scores,0);

		// sum up horizontal errors in the region
		for( int row = 0; row < regionHeight; row++ ) {
			// pixel indexes
			int startLeft = left.startIndex + left.stride*(y-radiusY+row) + x-radiusX;
			int startRight = right.startIndex + right.stride*(y-radiusY+row) + x-radiusX-minDisparity;

			for( int i = 0; i < localMaxDisparity; i++ ) {
				int indexLeft = startLeft;
				int indexRight = startRight-i;

				int score = 0;
				for( int j = 0; j < regionWidth; j++ ) {
					int diff = (left.data[ indexLeft++ ]& 0xFF) - (right.data[ indexRight++ ]& 0xFF);

					score += Math.abs(diff);
				}
				scores[i] += score;
			}
		}

		return true;
	}

	@Override
	public int[] getScore() {
		return scores;
	}

	@Override
	public Class<GrayU8> getImageType() {
		return GrayU8.class;
	}

}