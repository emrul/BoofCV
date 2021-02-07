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

package boofcv.examples.recognition;

import boofcv.abst.scene.ImageRecognition;
import boofcv.abst.scene.nister2006.ConfigImageRecognitionNister2006;
import boofcv.abst.scene.nister2006.ImageRecognitionNister2006;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageGray;
import boofcv.struct.image.ImageType;
import org.ddogleg.struct.DogArray;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * @author Peter Abeles
 **/
public class ExampleImageRecognition {

	private static class ImageIterator<T extends ImageGray<T>> implements Iterator<T> {

		T image;
		List<String> paths;
		int index = 0;

		public ImageIterator( List<String> paths, ImageType<T> imageType ) {
			image = imageType.createImage(1, 1);
			this.paths = paths;
		}

		@Override public boolean hasNext() {
			return index < paths.size();
		}

		@Override public T next() {
			BufferedImage buffered = UtilImageIO.loadImage(paths.get(index++));
			ConvertBufferedImage.convertFrom(buffered, true, image);
			return image;
		}
	}

	public static void main( String[] args ) {
		String imagePath = UtilIO.pathExample("recognition/vacation");
		List<String> images = UtilIO.listByPrefix(imagePath, null, ".jpg");
		Collections.sort(images);

		// Learn how to describe images
		var config = new ConfigImageRecognitionNister2006();
		config.tree.branchFactor = 4;
//		config.tree.maximumLevel = 4;
		ImageRecognition<GrayU8> recognizer = new ImageRecognitionNister2006<>(config, ImageType.SB_U8);
		recognizer.setVerbose(System.out, null);

		recognizer.learnDescription(new ImageIterator<>(images, ImageType.SB_U8));

		// Add images to the data base
		System.out.println("Adding images to the database");
		var iterator = new ImageIterator<>(images, ImageType.SB_U8);
		while (iterator.hasNext()) {
			recognizer.addDataBase(images.get(iterator.index), iterator.next());
		}

		// Look up images
		DogArray<ImageRecognition.Match> matches = new DogArray<>(ImageRecognition.Match::new);
		iterator = new ImageIterator<>(images, ImageType.SB_U8);
		while (iterator.hasNext()) {
			recognizer.findBestMatch(iterator.next(), matches);
			System.out.println(images.get(iterator.index-1)+" -> "+matches.get(0).id+"  matches.size="+matches.size);
		}

		System.out.println("Done");
	}
}
