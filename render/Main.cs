using ILGPU;
using ILGPU.Runtime;
using Library;
using Microsoft.CodeAnalysis.CodeStyle;
using MyApp;
using ScottPlot.Avalonia;
//main endpoint for program
public static class Main
{
    static string[] derivatives = new[]{
        "t+v1_xx-v0_xxy+t*mu", //v0_t=
        "v1_x-t*v0_xyy-t*ro", //v1_t=
    };
    static string[] partial = ["v1_xx", "v0_xxy", "v1_x", "v0_xyy"];
    static string[] constants = ["mu", "ro"];
    static double[] constantsValues = [1, 1];
    static double h = 0.01;
    static int xsize = 100;
    static int ysize = 100;
    //step size
    static double dt = 0.001;
    public static async Task Run(MainWindow window)
    {
        await Task.Yield();
        System.Console.WriteLine("start");


        double[][,,] initialValues = [new double[xsize, ysize, 1], new double[xsize, ysize, 1]];

        foreach (var init in initialValues)
        {
            for (int i = 0; i < xsize; i++)
                for (int j = 0; j < ysize; j++)
                    init[i, j, 0] = Random.Shared.NextDouble();
        }
        foreach (var init in initialValues)
        {
            for (int i = 0; i < 20; i++)
                for (int j = 0; j < 30; j++)
                {
                    init[i + 20, j + 50, 0] = Random.Shared.NextDouble() * 30; ;
                }
        }
        var gpuSolver = new GpuDiffEqSystemSolver3D(derivatives, DerivativeMethod.RungeKutta, partial, constants);
        var cpuSolver = new CpuDiffEqSystemSolver3D(derivatives, DerivativeMethod.RungeKuttaCpu, partial, constants);

        var solver = gpuSolver;

        // choose different versions of derivative computation algorithms
        // IDiffEqSolver solver = cpuSolver;//gpu solver;

        solver.CompileKernel();

        var solutionsObj = solver.Solutions(initialValues, dt, 0, h, 1, 2, 3, constantsValues);
        var solutions = solutionsObj.EnumerateSolutions();
        solutions.First();//initialize 


        var data = new double[xsize, ysize];
        foreach (var s in solutions)
        {
            await Task.Delay(200);
            CopyAvg(s, data);
            window.ScottPlotRender(plt =>
            {
                plt.Clear();
                plt.Add.Heatmap(data);
                plt.Axes.SetLimitsX(0, data.GetLength(0));
                plt.Axes.SetLimitsY(0, data.GetLength(1));
            });
            System.Console.WriteLine(s.Time);
        }
    }

    private static void CopyAvg((ArrayView3D<double, Stride3D.DenseXY>[] Values, double Time) s, double[,] data)
    {
        var d1 = s.Values[0].ToCpu();
        var d2 = s.Values[1].ToCpu();
        for (int i = 0; i < data.GetLength(0); i++)
            for (int j = 0; j < data.GetLength(0); j++)
            {
                data[i, j] = d1[i, j, 0]+d2[i, j, 0];
            }
    }
}
