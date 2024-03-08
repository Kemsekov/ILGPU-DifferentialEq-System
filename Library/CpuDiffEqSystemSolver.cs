using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using static Library.DerivativeMethod;
namespace Library
{
    public class CpuDiffEqSystemSolver
    {
        private Dictionary<string, int> _constantNameToId;
        private int size;
        private DerivMethod derivativeMethod;
        private Lazy<Func<double, double[], double>[]> functions;

        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(Func<double, double[], double>[] derivatives, DerivMethod derivativeMethod)
        {
            _constantNameToId = new Dictionary<string, int>();
            size = derivatives.Length;
            functions = new Lazy<Func<double, double[], double>[]>(() => derivatives);
            this.derivativeMethod = derivativeMethod;
        }
        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(string[] derivatives, DerivMethod derivativeMethod, string[]? constants = null)
        {
            constants ??= new string[] { };
            _constantNameToId = constants.Select((i, v) => (i, v)).ToDictionary(v => v.i, v => v.v);
            derivatives = derivatives.Select(d =>
            {
                foreach (var c in _constantNameToId)
                {
                    d = d.Replace(c.Key, $"v[{size + c.Value}]");
                }
                return d;
            }).ToArray();
            
            size = derivatives.Length;
            var derivFunctions =
                derivatives.Select((v, i) => $"double f{i}(double t,double[] v)=>{v};")
                .ToArray();

            var funcDecl = derivatives.Select((v, i) => $"f{i}").ToArray();
            var code =
            @"
            System.Func<double,double[],double>[]
            Execute(int a)
            {
                " + string.Join("\n", derivFunctions) + @"
                return " + $"new System.Func<double,double[],double>[]{{ {string.Join(",", funcDecl)} }} ;" + @"
            }
            ";
            this.derivativeMethod = derivativeMethod;
            functions = new Lazy<Func<double, double[], double>[]>(() => DynamicCompilation.CompileFunction<int, Func<double, double[], double>[]>(code)(1));
        }
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        public void CompileKernel()
        {
            var _ = functions.Value;
        }
        public SolutionsCpu Solutions(double[] initialValues, double dt, double t0, double[]? constants = null)
        {
            //previous values of x,y,z...
            var P = new double[size + _constantNameToId.Count];
            Buffer.BlockCopy(initialValues, 0, P, 0, size * sizeof(double));
            //new values of x,y,z...
            var V = new double[size + _constantNameToId.Count];
            return new SolutionsCpu(size, _constantNameToId.Count, P, V, _Kernel, dt, t0, constants);
        }
        private void _Kernel(double t, double[] p, double[] v, double dt)
        {
            var funcs = functions.Value;
            Parallel.For(0, size, i => v[i] = derivativeMethod(p, dt, t, i, funcs[i]));
        }

    }
}