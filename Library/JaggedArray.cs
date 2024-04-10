using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Security.Principal;
using Array3DView=ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;
using Array3D=ILGPU.Runtime.MemoryBuffer3D<double, ILGPU.Stride3D.DenseXY>;

namespace Library
{
    /// <summary>
    /// 100 length fixed array of Array3DView
    /// </summary>
    public unsafe struct Jagged3D_100{
        const int MaxSize = 100;
        //Array3DView size is 56 bytes
        const int Array3DViewSize = 56;
        fixed byte InnerStructs[Array3DViewSize * MaxSize]; // An array of InnerUnmanagedStruct
        public Jagged3D_100(IEnumerable<Array3D> arrays) : this(arrays.Select(v=>v.View).ToArray()){}
        public Jagged3D_100(IEnumerable<Array3DView> arrays) : this(arrays.ToArray()){}
        public Jagged3D_100(Array3DView[] arrays)
        {
            if(arrays.Length>=MaxSize)
                throw new ArgumentException($"Cannot create jagged array with more than {MaxSize} elements. \nFor some reason ilgpu does not support structs with more than {MaxSize} fields.");
            for(int i = 0;i<MaxSize;i++)
                this[i]=arrays[i];
        }
        public ref Array3DView this[int index]
        {
            get{
                fixed (byte* ptr = InnerStructs)
                {
                    #pragma warning disable
                    return ref ((Array3DView*)ptr)[index];
                }
            }
        }   
    }
    
}