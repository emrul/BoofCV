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

package boofcv.factory.feature.dense;

/**
 * Specifies how the image should be sampled when computing dense features
 *
 * @author Peter Abeles
 */
public class DenseSampling {
	/**
	 * Sample period along x-axis in pixels
	 */
	public double periodX;
	/**
	 * Sample period along y-axis in pixels
	 */
	public double periodY;

	public DenseSampling(double periodX, double periodY) {
		this.periodX = periodX;
		this.periodY = periodY;
	}

	public DenseSampling() {
	}

	public void setTo(DenseSampling src) {
		this.periodX = src.periodX;
		this.periodY = src.periodY;
	}
}
