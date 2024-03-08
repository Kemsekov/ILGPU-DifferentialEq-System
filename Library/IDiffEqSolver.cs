using System;
using System.Collections.Generic;

namespace Library
{
    public interface IDiffEqSolver
    {
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        void CompileKernel();
        IEnumerable<(double[] Values, double Time)> EnumerateSolutions(double[] initialValues, double dt, double t0,double[]? constants = null);
    }
    public interface IDiffEqSolver3D
    {
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        void CompileKernel();
        IEnumerable<(double[][,,] Values, double Time)> EnumerateSolutions(double[][,,] initialValues, double dt, double t0, double h, double x0, double y0, double z0, double[]? constants = null);
    }
}
