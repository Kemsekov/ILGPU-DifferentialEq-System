using System;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;

namespace Library
{
    public static class ArrayViewExtensions
    {
        public static double[,,] ToCpu(this Array3DView view){
            var data = new double[view.Extent.X,view.Extent.Y,view.Extent.Z];
            view.CopyToCPU(data);
            return data;
        }
        public static double[] ToCpu(this ArrayView1D<double, Stride1D.Dense> view){
            var data = new double[view.Extent.X];
            view.CopyToCPU(data);
            return data;
        }
    }
}
