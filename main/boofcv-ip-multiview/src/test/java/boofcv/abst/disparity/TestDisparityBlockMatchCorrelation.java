/*
 * Copyright (c) 2020, Peter Abeles. All Rights Reserved.
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

package boofcv.abst.disparity;

import boofcv.factory.disparity.ConfigDisparityBM;
import boofcv.factory.disparity.DisparityError;
import boofcv.factory.disparity.FactoryStereoDisparity;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageType;
import boofcv.testing.BoofStandardJUnit;
import org.junit.jupiter.api.Nested;

/**
 * @author Peter Abeles
 */
class TestDisparityBlockMatchCorrelation extends BoofStandardJUnit {
	@Nested
	class NCC_F32_U8 extends GenericStereoDisparityChecks<GrayF32, GrayU8> {

		public NCC_F32_U8() {
			super(ImageType.SB_F32, ImageType.SB_U8);
		}

		@Override
		public StereoDisparity<GrayF32, GrayU8> createAlg( int disparityMin, int disparityRange ) {
			ConfigDisparityBM config = new ConfigDisparityBM();
			config.errorType = DisparityError.NCC;
			config.subpixel = false;
			config.regionRadiusX = config.regionRadiusY = 1;
			config.disparityMin = disparityMin;
			config.disparityRange = disparityRange;
			return FactoryStereoDisparity.blockMatch(config, inputType.getImageClass(), disparityType.getImageClass());
		}
	}

	@Nested
	class NCC_F32_F32 extends GenericStereoDisparityChecks<GrayF32, GrayF32> {

		public NCC_F32_F32() {
			super(ImageType.SB_F32, ImageType.SB_F32);
		}

		@Override
		public StereoDisparity<GrayF32, GrayF32> createAlg( int disparityMin, int disparityRange ) {
			ConfigDisparityBM config = new ConfigDisparityBM();
			config.errorType = DisparityError.NCC;
			config.subpixel = true;
			config.regionRadiusX = config.regionRadiusY = 1;
			config.disparityMin = disparityMin;
			config.disparityRange = disparityRange;
			return FactoryStereoDisparity.blockMatch(config, inputType.getImageClass(), disparityType.getImageClass());
		}
	}
}
