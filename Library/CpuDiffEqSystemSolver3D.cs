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
    public class CpuDiffEqSystemSolver3D
    {
        private int size;
        private int derivativesSize;
        private DerivMethod derivativeMethod;
        /// <summary>
        /// f(double t,double[] v);
        /// where t is time, v is variables that have size elements at the
        /// beginning as variables and others as derivatives
        /// </summary>
        private Lazy<Func<double, double[], double>[]> functions;
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

        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        /// <param name="partialDerivatives">
        /// A list of used derivatives in a form v[number]_[x|y|z]+; for example v0_xy is xy derivative of first variable that is v[0]. <br/>
        /// Higher order derivatives must be written in increasing order [x,y,z]. <br/>
        /// v0_xxyzz - valid <br/>
        /// v0_zyxz - invalid <br/>
        /// </param>
        public CpuDiffEqSystemSolver3D(string[] derivatives, DerivMethod derivativeMethod, string[] partialDerivatives)
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
                if(derivatives!=split[1]){
                    throw new ArgumentException("Partial derivatives must follow the rule of increasing. v0_xxyz is valid, v0_yxzx is invalid");
                }
                _derivIndexMap[$"v{variable}_{derivatives}"] = i;
                _indexDerivMap[i] = (variable, derivatives);

                return (variable, derivatives);
            })
            .SelectMany(v => Enumerable.Range(1, v.derivatives.Length).Select(i => (v.variable, v.derivatives[..i])))
            .GroupBy(v => v.variable)
            .ToDictionary(v => v.First().variable, v => v.Select(t => t.Item2).OrderBy(v => v.Length).ToArray());

            derivativesSize=_derivIndexMap.Count;
            //replace all `v<i>_[x|y|z]+` to `v[v.Length+m]` with unique mapping
            derivatives = derivatives.Select(d =>
            {
                foreach (var pair in _derivIndexMap)
                {
                    d = d.Replace(pair.Key, $"v[{pair.Value+size}]");
                }
                d = d.Replace("X", $"v[{^1}]");
                d = d.Replace("Y", $"v[{^2}]");
                d = d.Replace("Z", $"v[{^3}]");
                d = d.Replace("Math","System.Math");
                return d;
            }).ToArray();

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
        /// <param name="initialValues">3d grid initial values for each variable</param>
        /// <param name="dt">time step size</param>
        /// <param name="t0">time zero</param>
        /// <param name="h">grid step size</param>
        /// <returns></returns>
        public IEnumerable<(double[][,,] Values, double Time)> EnumerateSolutions(double[][,,] initialValues, double dt, double t0, double h,double x0,double y0,double z0)
        {
            var size0 = initialValues[0].GetLength(0);
            var size1 = initialValues[0].GetLength(1);
            var size2 = initialValues[0].GetLength(2);
            //previous values of x,y,z...
            var P = initialValues;
            //new values of x,y,z...
            var V = initialValues.Select(grid => new double[grid.GetLength(0), grid.GetLength(1), grid.GetLength(2)]).ToArray();

            yield return (P, t0);

            //derivatives grid
            //so grid for derivative v1_xy can be found at
            //derivatives[1].Find(v=>v.derivative=="xy")
            Dictionary<int, (string derivative, double[,,] grid)[]> derivatives = _partial.ToDictionary(a => a.Key, a => a.Value.Select(derivative => (derivative, new double[size0, size1, size2])).ToArray());

            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t, P, V, dt, derivatives, h,x0,y0,z0);
                yield return (V, t);
                (P, V) = (V, P);
            }
        }

        private void _Kernel(double t, double[][,,] p, double[][,,] v, double dt, Dictionary<int, (string derivative, double[,,] grid)[]> derivatives, double h,double x0,double y0,double z0)
        {

            var derivsPlaced = new double[derivativesSize][,,];
            var axisDeriv = new Dictionary<string,double[,,]>();
            //for all required derivatives compute their grid values
            foreach (var (variable, derivs) in derivatives)
            {
                var previous = p[variable];
                foreach (var partial in derivs)
                {
                    if(_derivIndexMap.TryGetValue($"v{variable}_{partial.derivative}",out var i)){
                        derivsPlaced[i]=partial.grid;
                    }
                    var axis = partial.derivative.Last();
                    if(partial.derivative.Length==1)
                        previous = p[variable];
                    else
                        previous = axisDeriv[partial.derivative[..^1]];
                    switch (char.ToLower(axis))
                    {
                        case 'x':
                            GridDerivativeKernelX(previous, h, partial.grid);
                            break;
                        case 'y':
                            GridDerivativeKernelY(previous, h, partial.grid);
                            break;
                        case 'z':
                            GridDerivativeKernelZ(previous, h, partial.grid);
                            break;
                    }
                    axisDeriv[partial.derivative]=partial.grid;
                }
            }
            
            var funcs = functions.Value;
            //for each variable update it's grid
            for(int i = 0;i<funcs.Length;i++)
                VariableGridUpdateKernel(i,p,v,dt,t,derivsPlaced,h,x0,y0,z0);
        }

        private void VariableGridUpdateKernel(int variableIndex,double[][,,] p, double[][,,] v, double dt,double t, double[][,,] derivsPlaced,double h,double x0,double y0,double z0)
        {
            var size0 = p[0].GetLength(0);
            var size1 = p[0].GetLength(1);
            var size2 = p[0].GetLength(2);
            var funcs = functions.Value;
            
            var input = new double[size+derivativesSize+3];
            var output = new double[size];
            // Parallel.For(0,size0,i=>{
                for (int i = 0; i < size0; i++)
                for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++){
                    //pass previous values
                    for(int w = 0;w<size;w++)
                        input[w]=p[w][i,j,k];
                    for(int w = 0;w<derivativesSize;w++)
                        input[w+size]=derivsPlaced[w][i,j,k];
                    input[^1]=i*h+x0;
                    input[^2]=j*h+y0;
                    input[^3]=k*h+z0;
                    //compute new values
                    derivativeMethod(input,output,dt,t,variableIndex,funcs[variableIndex]);

                    //save new values
                    v[variableIndex][i,j,k]=output[variableIndex];
                }
            // });
        }

        private void GridDerivativeKernelZ(double[,,] previous, double h, double[,,] grid)
        {
            var size0 = grid.GetLength(0);
            var size1 = grid.GetLength(1);
            var size2 = grid.GetLength(2);
            Parallel.For(0,size0,i=>{
                for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++){
                    grid[i,j,k]=DerivativeMethod.DerivativeZ(i,j,k,h,previous);
                }
            });
        }

        private void GridDerivativeKernelY(double[,,] previous, double h, double[,,] grid)
        {
            var size0 = grid.GetLength(0);
            var size1 = grid.GetLength(1);
            var size2 = grid.GetLength(2);
            Parallel.For(0,size0,i=>{
                for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++){
                    grid[i,j,k]=DerivativeMethod.DerivativeY(i,j,k,h,previous);
                }
            });
        }

        private void GridDerivativeKernelX(double[,,] previous, double h, double[,,] grid)
        {
            var size0 = grid.GetLength(0);
            var size1 = grid.GetLength(1);
            var size2 = grid.GetLength(2);
            Parallel.For(0,size0,i=>{
                for (int j = 0; j < size1; j++)
                for (int k = 0; k < size2; k++){
                    grid[i,j,k]=DerivativeMethod.DerivativeX(i,j,k,h,previous);
                }
            });
        }
    }
}