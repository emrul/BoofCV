/*
 * Copyright (c) 2011, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://www.boofcv.org).
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

package boofcv.abst.feature.detect.line;

import boofcv.alg.filter.derivative.GImageDerivativeOps;
import boofcv.factory.feature.detect.line.FactoryDetectLine;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageFloat32;
import boofcv.struct.image.ImageUInt8;


/**
 * @author Peter Abeles
 */
public class TestDetectLineHoughPolar extends GeneralDetectLineTests {


	public TestDetectLineHoughPolar() {
		super(ImageUInt8.class,ImageFloat32.class);
	}

	@Override
	public <T extends ImageBase>
	DetectLine<T> createAlg(Class<T> imageType) {

		Class derivType = GImageDerivativeOps.getDerivativeType(imageType);

		return FactoryDetectLine.houghPolar(2, 3, 40, 180, 10, 20 , imageType, derivType);
	}
}
