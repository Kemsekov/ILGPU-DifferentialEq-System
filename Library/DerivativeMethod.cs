using System;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;

namespace Library
{
    public class DerivativeMethod
    {
        public delegate double DerivMethod(double[] prev, double dt, double t, int i, Func<double, double[], double> f_i);
        public const string Euler = "newV[i] = prev[i] + dt * f<i>(t,prev);";
        public static double EulerCpu(double[] prev, double dt, double t, int i, Func<double, double[], double> f_i)
        {
            return prev[i] + dt * f_i(t, prev);
        }
        public const string ImprovedEuler =
        @"
        tmp3=prev[i];
        tmp1=f<i>(t,prev);
        prev[i]=tmp3 + dt * tmp1;
        newV[i]=tmp3+0.5*dt*(tmp1+f<i>(t+dt,prev));
        prev[i]=tmp3;
        ";
        public static double ImprovedEulerCpu(double[] prev, double dt, double t, int i, Func<double, double[], double> f_i)
        {
            var originalPrev = prev[i];
            var deriv = f_i(t, prev);
            prev[i] = originalPrev + dt * deriv;
            var res = originalPrev + 0.5f * dt * (deriv + f_i(t + dt, prev));
            prev[i] = originalPrev;
            return res;
        }
        public const string RungeKutta =
        @"
        tmp1=dt/2;
        tmp6 = prev[i];
        tmp2 = f<i>(t,prev);
        prev[i]=tmp6+tmp1*tmp2;
        tmp3 = f<i>(t+tmp1,prev);
        prev[i]=tmp6+tmp1*tmp3;
        tmp4 = f<i>(t+tmp1,prev);
        prev[i]=tmp6+dt*tmp4;
        tmp5 = f<i>(t+dt,prev);
        prev[i]=tmp6;
        newV[i] = tmp6+dt/6*(tmp2+2*tmp3+2*tmp4+tmp5);
        ";
        public static double RungeKuttaCpu(double[] prev, double dt, double t, int i, Func<double, double[], double> f_i)
        {
            var dtHalf = dt / 2;
            var originalPrev = prev[i];
            var k1 = f_i(t, prev);
            prev[i] = originalPrev + dtHalf * k1;
            var k2 = f_i(t + dtHalf, prev);
            prev[i] = originalPrev + dtHalf * k2;
            var k3 = f_i(t + dtHalf, prev);
            prev[i] = originalPrev + dt * k3;
            var k4 = f_i(t + dt, prev);
            prev[i] = originalPrev;
            return originalPrev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        /// <summary>
        /// Approximates first derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="by">Dimension to take derivative of</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public double Derivative(int i, int j, int k, int by, double h, double[,,] grid)
        {
            switch (by)
            {
                case 0:
                    return DerivativeX(i, j, k, h, grid);
                case 1:
                    return DerivativeY(i, j, k, h, grid);
                case 2:
                    return DerivativeZ(i, j, k, h, grid);
            }
            throw new ArgumentException("Cannot find derivative at index " + by + " for 3-dimensional tensor");
        }
        /// <summary>
        /// Approximates first z derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeZ(int i, int j, int k, double h, double[,,] grid)
        {
            int size2 = grid.GetLength(2);
            return CentralDifference4Order(
                h,
                grid[i, j, (k + 1)%size2],
                grid[i, j, (k + 2)%size2],
                grid[i, j, (size2+k - 1)%size2],
                grid[i, j, (size2+k - 2)%size2]
            );
        }

        /// <summary>
        /// Approximates first y derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeY(int i, int j, int k, double h, double[,,] grid)
        {
            int size1 = grid.GetLength(1);
            return CentralDifference4Order(
                h,
                grid[i, (j + 1) % size1, k],
                grid[i, (j + 2) % size1, k],
                grid[i, (size1 + j - 1) % size1, k],
                grid[i, (size1 + j - 2) % size1, k]
            );

        }

        /// <summary>
        /// Approximates first x derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeX(int i, int j, int k, double h, double[,,] grid)
        {
            var size0 = grid.GetLength(0);
            return CentralDifference4Order(
                h,
                grid[(i + 1) % size0, j, k],
                grid[(i + 2) % size0, j, k],
                grid[(size0 + i - 1) % size0, j, k],
                grid[(size0 + i - 2) % size0, j, k]
            );

        }

        /// <summary>
        /// Approximates first z derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeZ(int i, int j, int k, double h, Array3DView grid)
        {
            long size2 = grid.Extent.Y;
            return CentralDifference4Order(
                h,
                grid[i, j, (k + 1) % size2],
                grid[i, j, (k + 2) % size2],
                grid[i, j, (size2 + k - 1) % size2],
                grid[i, j, (size2 + k - 2) % size2]
            );

        }

        /// <summary>
        /// Approximates first y derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeY(int i, int j, int k, double h, Array3DView grid)
        {
            long size1 = grid.Extent.Y;
            return CentralDifference4Order(
                h,
                grid[i, (j + 1) % size1, k],
                grid[i, (j + 2) % size1, k],
                grid[i, (size1 + j - 1) % size1, k],
                grid[i, (size1 + j - 2) % size1, k]
            );

        }

        /// <summary>
        /// Approximates first x derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static double DerivativeX(int i, int j, int k, double h, Array3DView grid)
        {
            var size0 = grid.Extent.Y;
            return CentralDifference4Order(
                h,
                grid[(i + 1) % size0, j, k],
                grid[(i + 2) % size0, j, k],
                grid[(size0 + i - 1) % size0, j, k],
                grid[(size0 + i - 2) % size0, j, k]
            );

        }

        static double BackwardDifference3Points(double h, double u, double u_m, double u_mm)
        {
            return (3 * u - 4 * u_m + u_mm) * 0.5f / h;
        }
        static double ForwardDifference3Points(double h, double u, double u_p, double u_pp)
        {
            return (-3 * u + 4 * u_p - u_pp) * 0.5f / h;
        }
        static double CentralDifference(double h, double u_p, double u_m)
        {
            return (u_p - u_m) * 0.5f / h;
        }
        static double CentralDifference4Order(double h, double u_p, double u_pp, double u_m, double u_mm)
        {
            return 0.0833333333f / h * (8 * (u_p - u_m) + u_mm - u_pp);
        }
    }
}