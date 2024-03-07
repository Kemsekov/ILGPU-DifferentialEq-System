using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;
using Array3D = ILGPU.Runtime.MemoryBuffer3D<double, ILGPU.Stride3D.DenseXY>;
using static Library.DerivativeMethod;
using GridDerivativeMethod=System.Action<ILGPU.Index1D, float, ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>, double, ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>>;
using VariableGridUpdateKernelMethod = System.Action<ILGPU.Index2D, Library.Jagged3D_10, Library.Jagged3D_10, Library.Jagged3D_60, double, double, double, double, double, double>;
namespace Library
{
    public class GpuDiffEqSystemSolver3D : IDisposable
    {
        private int size;
        private int derivativesSize;
        /// <summary>
        ///if given v0_xxy then
        ///_partial[0]=["x","xx","xxy"]
        /// </summary>
        private Dictionary<int, string[]> _partial;
        /// <summary>
        /// map each deriv
        /// v[i]_[x|y|z]+ to unique number
        /// </summary>
        private Dictionary<string, int> _derivIndexMap;
        /// <summary>
        /// backwards map unique number to deriv v[i]_[x|y|z]+
        /// </summary>
        private Dictionary<int, (int i, string derivatives)> _indexDerivMap;
        private Lazy<VariableGridUpdateKernelMethod> loadedKernel;
        private Lazy<GridDerivativeMethod> gridDerivativeKernel;
        private Context context;
        private Accelerator accelerator;

        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        /// <param name="partialDerivatives">
        /// A list of used derivatives in a form v[number]_[x|y|z]+; for example v0_xy is xy derivative of first variable that is v[0]. <br/>
        /// Higher order derivatives must be written in increasing order [x,y,z]. <br/>
        /// v0_xxyzz - valid <br/>
        /// v0_zyxz - invalid <br/>
        /// </param>
        public GpuDiffEqSystemSolver3D(string[] derivatives, string derivativeMethod, string[] partialDerivatives)
        {
            size = derivatives.Length;
            // map each deriv
            // v<i>_[x|y|z]+ to unique number
            _derivIndexMap = new Dictionary<string, int>();
            // backwards map unique number to deriv v<i>_[x|y|z]+
            _indexDerivMap = new Dictionary<int, (int i, string derivatives)>();

            //if given v0_xxy then
            //_partial[0]=["x","xx","xxy"]
            _partial =
            partialDerivatives.Select((d, i) =>
            {
                var split = d.Split("_");
                var variable = int.Parse(new string(split[0][1..]));
                var derivatives = new string(split[1].OrderBy(c => (int)c).ToArray());
                if (derivatives != split[1])
                {
                    throw new ArgumentException("Partial derivatives must follow the rule of increasing. v0_xxyz is valid, v0_yxzx is invalid");
                }
                _derivIndexMap[$"v{variable}_{derivatives}"] = i;
                _indexDerivMap[i] = (variable, derivatives);

                return (variable, derivatives);
            })
            .SelectMany(v => Enumerable.Range(1, v.derivatives.Length).Select(i => (v.variable, v.derivatives[..i])))
            .GroupBy(v => v.variable)
            .ToDictionary(v => v.First().variable, v => v.Select(t => t.Item2).OrderBy(v => v.Length).ToArray());

            derivativesSize = _derivIndexMap.Count;
            //replace all `v<i>_[x|y|z]+` to `v[v.Length+m]` with unique mapping
            derivatives = derivatives.Select(d =>
            {
                foreach (var pair in _derivIndexMap)
                {
                    d = d.Replace(pair.Key, $"v[{pair.Value + size}]");
                }
                d = d.Replace("X", $"v[{^1}]");
                d = d.Replace("Y", $"v[{^2}]");
                d = d.Replace("Z", $"v[{^3}]");
                d = d.Replace("Math", "System.Math");
                return d;
            }).ToArray();

            var derivFunctions =
                derivatives.Select((v, i) => $"double f{i}(double t,double[] v)=>{v};")
                .ToArray();

            var derivCases =
                Enumerable.Range(0, size)
                .Select(i => $"case {i}: {derivativeMethod.Replace("<i>", i.ToString())}; break;")
                .ToArray();
            var kernelCode =
            @"static void Kernel_(int i,double t,double dt, double[] prev, double[] newV){ 
            " + string.Join("\n", derivFunctions) + @"
                //temp values used for computation
                double tmp1 = 0,tmp2 = 0,tmp3 = 0,tmp4 = 0,tmp5 = 0,tmp6 = 0; 
                switch(i){
            " + string.Join("\n", derivCases) + @"

                }
            }";

            var updateVariableMethod =
            @"static void VariableGridUpdateKernel(ILGPU.Index2D point, Library.Jagged3D_10 p, Library.Jagged3D_10 v, Library.Jagged3D_60 derivsPlaced, double dt, double t, double h, double x0, double y0, double z0)
            {
            //x is grid x coordinate
            //i is variable index v0, v1 ...
            var x = (int)point.X;
            var i = (int)point.Y;

            var size1 = p[0].Extent.Y;
            var size2 = p[0].Extent.Z;
            var size = "+size+@";
            var derivativesSize = "+derivativesSize+@";

            var prev = new double[" + size + derivativesSize + 3 + @"];
            var newV = new double[" + (size+1) + @"];
            for (int j = 0; j < size1; j++)
            for (int k = 0; k < size2; k++)
            {
                //pass previous values
                for (int w = 0; w < size; w++)
                    prev[w] = p[w][x, j, k];
                for (int w = 0; w < derivativesSize; w++)
                    prev[w + size] = derivsPlaced[w][x, j, k];
                prev[^1] = x * h + x0;
                prev[^2] = j * h + y0;
                prev[^3] = k * h + z0;

                //compute new values
                Kernel_(i,t,dt,prev,newV);

                v[i][x, j, k] = newV[i];
            }
            }";
            var code =
            @"
            System.Action<ILGPU.Index2D, Library.Jagged3D_10, Library.Jagged3D_10, Library.Jagged3D_60, double, double, double, double, double, double>
            Execute(ILGPU.Runtime.Accelerator accelerator)
            {
                return ILGPU.Runtime.KernelLoaders
                    .LoadAutoGroupedStreamKernel<ILGPU.Index2D, Library.Jagged3D_10, Library.Jagged3D_10, Library.Jagged3D_60, double, double, double, double, double, double>
                        (accelerator,VariableGridUpdateKernel);
            }
            " + kernelCode + updateVariableMethod;

            context = Context.CreateDefault();
            accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);

            loadedKernel = new Lazy<VariableGridUpdateKernelMethod>(() => DynamicCompilation.CompileFunction<Accelerator, VariableGridUpdateKernelMethod>(
                code,
                typeof(Index1D), typeof(ArrayView<int>), typeof(KernelLoaders), typeof(Jagged3D_10), typeof(Jagged3D_60))
            (accelerator));
            gridDerivativeKernel = new Lazy<GridDerivativeMethod>(()=>accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index1D, float, Array3DView, double, Array3DView>(GridDerivativeKernel));
            
        }

        /// <summary>
        /// Precompiles kernel
        /// </summary>
        public void CompileKernel()
        {
            var _ = loadedKernel.Value;
        }
        /// <param name="initialValues">3d grid initial values for each variable</param>
        /// <param name="dt">time step size</param>
        /// <param name="t0">time zero</param>
        /// <param name="h">grid step size</param>
        /// <returns></returns>
        public IEnumerable<(double[][,,] Values, double Time)> EnumerateSolutions(double[][,,] initialValues, double dt, double t0, double h, double x0, double y0, double z0)
        {
            var size0 = initialValues[0].GetLength(0);
            var size1 = initialValues[0].GetLength(1);
            var size2 = initialValues[0].GetLength(2);

            var buffer = initialValues.Select(grid => new double[grid.GetLength(0), grid.GetLength(1), grid.GetLength(2)]).ToArray();

            //previous values of x,y,z...

            var P = initialValues.Select(grid => accelerator.Allocate3DDenseXY(grid)).ToArray();
            //new values of x,y,z...
            var V = initialValues.Select(grid => accelerator.Allocate3DDenseXY<double>((grid.GetLength(0), grid.GetLength(1), grid.GetLength(2)))).ToArray();
            Move(buffer, P);
            yield return (buffer, t0);

            //derivatives grid
            //so grid for derivative v1_xy can be found at
            //derivatives[1].Find(v=>v.derivative=="xy")
            Dictionary<int, (string derivative, Array3D grid)[]> derivatives = _partial.ToDictionary(a => a.Key, a => a.Value.Select(derivative => (derivative, accelerator.Allocate3DDenseXY<double>((size0, size1, size2)))).ToArray());

            var pJagged = new Jagged3D_10(P);
            var vJagged = new Jagged3D_10(V);
            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t, pJagged, vJagged, dt, derivatives, h, x0, y0, z0);
                Move(buffer, V);
                yield return (buffer, t);
                (pJagged, vJagged) = (vJagged, pJagged);
            }
        }

        private static void Move(double[][,,] buffer, Array3D[] P)
        {
            for (int i = 0; i < P.Length; i++)
            {
                P[i].CopyToCPU(buffer[i]);
            }
        }

        private void _Kernel(double t, Jagged3D_10 p, Jagged3D_10 v, double dt, Dictionary<int, (string derivative, Array3D grid)[]> derivatives, double h, double x0, double y0, double z0)
        {

            //for each variable update it's grid
            var size0 = p[0].Extent.X;

            Array3D[] derivsPlaced = new Array3D[derivativesSize];
            var axisDeriv = new Dictionary<string, Array3D>();
            //for all required derivatives compute their grid values
            foreach (var (variable, derivs) in derivatives)
            {
                var previous = p[variable];
                foreach (var partial in derivs)
                {
                    if (_derivIndexMap.TryGetValue($"v{variable}_{partial.derivative}", out var i))
                    {
                        derivsPlaced[i] = partial.grid;
                    }
                    var axis = partial.derivative.Last();
                    if (partial.derivative.Length == 1)
                        previous = p[variable];
                    else
                        previous = axisDeriv[partial.derivative[..^1]];
                    var axisN = axis=='y' ? 1 : (axis=='z' ? 2 : 0);
                    gridDerivativeKernel.Value((Index1D)size0,axisN, previous, h, partial.grid);
                    axisDeriv[partial.derivative] = partial.grid;
                }
            }
            loadedKernel.Value(new Index2D((int)size0, size), p, v, new Jagged3D_60(derivsPlaced), dt, t, h, x0, y0, z0);
        }

        delegate void GridDerivativeKernelMethod(Index1D i, Array3DView previous, double h, Array3DView grid);

        static void GridDerivativeKernel(Index1D i, float by, Array3DView previous, double h, Array3DView grid)
        {
            switch (by)
            {
                case 0:
                    GridDerivativeKernelX(i, previous, h, grid);
                    break;
                case 1:
                    GridDerivativeKernelY(i, previous, h, grid);
                    break;
                case 2:
                    GridDerivativeKernelZ(i, previous, h, grid);
                    break;
            }
        }
        static void GridDerivativeKernelZ(Index1D i, Array3DView previous, double h, Array3DView grid)
        {
            var size1 = previous.Extent.Y;
            var size2 = previous.Extent.Z;
            for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++)
                {
                    grid[i, j, k] = DerivativeMethod.DerivativeZ(i, j, k, h, previous);
                }
        }

        static void GridDerivativeKernelY(Index1D i, Array3DView previous, double h, Array3DView grid)
        {
            var size1 = previous.Extent.Y;
            var size2 = previous.Extent.Z;
            for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++)
                {
                    grid[i, j, k] = DerivativeMethod.DerivativeY(i, j, k, h, previous);
                }
        }

        static void GridDerivativeKernelX(Index1D i, Array3DView previous, double h, Array3DView grid)
        {
            var size1 = previous.Extent.Y;
            var size2 = previous.Extent.Z;
            for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++)
                {
                    grid[i, j, k] = DerivativeMethod.DerivativeX(i, j, k, h, previous);
                }
        }

        public void Dispose()
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }
}