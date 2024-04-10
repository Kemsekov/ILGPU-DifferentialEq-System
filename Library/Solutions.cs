using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using KernelType = System.Action<ILGPU.Index1D, double, double, ILGPU.ArrayView<double>, ILGPU.ArrayView<double>>;
using CpuKernelType=System.Action<double, double[], double[], double>;
using Cpu3DKernelType=System.Action<double, double[][,,], double[][,,],double[], double, System.Collections.Generic.Dictionary<int, (string derivative, double[,,] grid)[]>, double, double, double, double>;
using Gpu3DKernelType=System.Action<double, Library.Jagged3D_100, Library.Jagged3D_100, ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>, double, System.Collections.Generic.Dictionary<int, (string derivative, ILGPU.Runtime.MemoryBuffer3D<double, ILGPU.Stride3D.DenseXY> grid)[]>, double, double, double, double>;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;
using Array3D = ILGPU.Runtime.MemoryBuffer3D<double, ILGPU.Stride3D.DenseXY>;
using Array1DView = ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>;
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
    public class SolutionsCpu3D{
        public SolutionsCpu3D(Cpu3DKernelType kernel,Dictionary<int, (string derivative, double[,,] grid)[]> derivatives, double[][,,] P, double[][,,] V,double dt, double t0, double h,double x0,double y0,double z0,double[] constants){
            this.constants = constants;
            this.derivatives=derivatives;
            this.P=P;
            this.V=V;
            this._Kernel=kernel;;
            this.dt=dt;;
            this.t0=t0;;
            this.h=h;;
            this.x0=x0;;
            this.y0=y0;;
            this.z0=z0;;
        }

        private double[] constants;
        private Dictionary<int, (string derivative, double[,,] grid)[]> derivatives;
        public double[][,,] P;
        public double[][,,] V;
        private Cpu3DKernelType _Kernel;
        private double dt;
        private double t0;
        private double h;
        private double x0;
        private double y0;
        private double z0;

        public IEnumerable<(double[][,,] Values, double Time)> EnumerateSolutions(){
            yield return (P, t0);
            
            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t, P, V,constants, dt, derivatives, h,x0,y0,z0);
                yield return (V, t);
                (P, V) = (V, P);
            }
        }
        public void UpdateConstants(double[] constants){
            this.constants=constants;
        }
    }
    public class SolutionsGpu3D{
        public SolutionsGpu3D(Gpu3DKernelType kernel,Dictionary<int, (string derivative, Array3D grid)[]> derivatives, Array3D[] P, Array3D[] V,double dt, double t0, double h,double x0,double y0,double z0,Array1DView constants){
            this.constants = constants;
            this.derivatives=derivatives;
            this.P=P;
            this.V=V;
            this._Kernel=kernel;;
            this.dt=dt;;
            this.t0=t0;;
            this.h=h;;
            this.x0=x0;;
            this.y0=y0;;
            this.z0=z0;;
        }

        private Array1DView constants;
        private Dictionary<int, (string derivative, Array3D grid)[]> derivatives;

        public Array3D[] P { get; }
        public Array3D[] V { get; }

        private Gpu3DKernelType _Kernel;
        private double dt;
        private double t0;
        private double h;
        private double x0;
        private double y0;
        private double z0;

        public IEnumerable<(Array3DView[] Values, double Time)> EnumerateSolutions(){
            yield return (P.Select(v=>v.View).ToArray(), t0);


            var pJagged = new Jagged3D_100(P);
            var vJagged = new Jagged3D_100(V);
            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t, pJagged, vJagged,constants, dt, derivatives, h, x0, y0, z0);
                yield return (V.Select(v=>v.View).ToArray(), t);
                (pJagged, vJagged) = (vJagged, pJagged);
            }
        }
        public void UpdateConstants(double[] constants){
            this.constants.CopyFromCPU(constants);
        }
    }
}