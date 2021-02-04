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

package boofcv.struct.kmeans;

import boofcv.struct.feature.TupleDesc_F64;
import org.ddogleg.struct.DogArray_F64;
import org.ddogleg.struct.LArrayAccessor;

/**
 * @author Peter Abeles
 */
public class PackedTupleArray_F64 implements LArrayAccessor<TupleDesc_F64> {
	// degree-of-freedom, number of elements in the tuple
	public final int dof;
	// Stores tuple in a single continuous array
	public final DogArray_F64 array;
	// tuple that the result is temporarily written to
	public final TupleDesc_F64 temp;

	// Number of tuples stored in the array
	protected int arraySize;

	public PackedTupleArray_F64( int dof ) {
		this.dof = dof;
		this.temp = new TupleDesc_F64(dof);
		array = new DogArray_F64(dof*20);
		array.resize(0);
	}

	public void reset() {
		arraySize = 0;
		array.reset();
	}

	public void reserve( int numTuples ) {
		array.reserve(numTuples*dof);
	}

	public void add( TupleDesc_F64 tuple ) {
		array.addAll(tuple.value, 0, dof);
	}

	@Override public TupleDesc_F64 getTemp( int index ) {
		System.arraycopy(array.data,index*dof,temp.value,0,dof);
		return temp;
	}

	@Override public void getCopy( int index, TupleDesc_F64 dst ) {
		System.arraycopy(array.data,index*dof,dst.value,0,dof);
	}

	@Override public void copy( TupleDesc_F64 src, TupleDesc_F64 dst ) {
		System.arraycopy(src.value,0,dst.value,0,dof);
	}

	@Override public int size() {
		return arraySize;
	}
}
