using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using KernelType = System.Action<ILGPU.Index1D, double, double, ILGPU.ArrayView<double>, ILGPU.ArrayView<double>>;
using CpuKernelType=System.Action<double, double[], double[], double>;
using System;
namespace Library
{
    public class SolutionsCpu{
        private int size;
        private double dt;
        private double t0;
        private int constantSize;

        public double[] P;
        public double[] V;

        private CpuKernelType kernel;

        public SolutionsCpu(int size, int constantSize,double[] P,double[] V, CpuKernelType kernel, double dt, double t0,double[]? constants= null){
            this.size=size;
            this.dt=dt;
            this.t0=t0;
            this.constantSize=constantSize;
            this.P=P;
            this.V=V;
            this.kernel=kernel;
            if(constants != null)
                UpdateConstants(constants);
        }
        public IEnumerable<(double[] Values, double Time)> EnumerateSolutions()
        {
            yield return (P, t0);
            
            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                kernel(t, P, V,dt);
                yield return (V, t);
                (P, V) = (V, P);
            }
        }
        public void UpdateConstants(double[] constants){
            Buffer.BlockCopy(constants,0,P,size*sizeof(double),constantSize*sizeof(double));
            Buffer.BlockCopy(constants,0,V,size*sizeof(double),constantSize*sizeof(double));
        }
    }
    public class SolutionsGpu{
        private Accelerator accelerator;
        private int size;
        private double dt;
        private double t0;
        private int constantSize;

        public MemoryBuffer1D<double, Stride1D.Dense> P;
        public MemoryBuffer1D<double, Stride1D.Dense> V;

        private KernelType kernel;

        public SolutionsGpu(int size, int constantSize,Accelerator accelerator,MemoryBuffer1D<double, Stride1D.Dense> P,MemoryBuffer1D<double, Stride1D.Dense> V, KernelType kernel, double dt, double t0,double[]? constants= null){
            this.accelerator=accelerator;
            this.size=size;
            this.dt=dt;
            this.t0=t0;
            this.constantSize=constantSize;
            this.P=P;
            this.V=V;
            this.kernel=kernel;
            if(constants != null)
                UpdateConstants(constants);
        }
        public IEnumerable<(ArrayView1D<double, Stride1D.Dense> Values, double Time)> EnumerateSolutions()
        {
            yield return (P, t0);
            
            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                kernel((Index1D)size, t, dt, P.View, V.View);
                accelerator.Synchronize();
                yield return (V.View, t);
                (P, V) = (V, P);
            }
        }
        public void UpdateConstants(double[] constants){
            P.View.SubView(size,constantSize).CopyFromCPU(constants);
            V.View.SubView(size,constantSize).CopyFromCPU(constants);
        }
    }
}