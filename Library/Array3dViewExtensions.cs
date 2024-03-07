using System;
using ILGPU.Runtime;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;

namespace Library
{
    public static class Array3dViewExtensions
    {
        public static double[,,] ToCpu(this Array3DView view){
            var data = new double[view.Extent.X,view.Extent.Y,view.Extent.Z];
            view.CopyToCPU(data);
            return data;
        }
    }
}
