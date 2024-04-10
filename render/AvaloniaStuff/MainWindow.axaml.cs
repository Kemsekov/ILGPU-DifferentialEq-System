using Avalonia.Controls;
using Avalonia.Media;
using ScottPlot.Avalonia;

namespace MyApp;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Main.Run(this);
    }
    public override void EndInit()
    {
        base.EndInit();
    }
    /// <summary>
    /// Renders whatever you do to plot object
    /// </summary>
    public void ScottPlotRender(Action<ScottPlot.Plot> plot){
        var avaPlot1 = this.Find<AvaPlot>("AvaPlot");
        if (avaPlot1 is not null)
        {
            plot(avaPlot1.Plot);
            avaPlot1.Refresh();
        }
    }
    
}