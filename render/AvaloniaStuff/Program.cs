using Avalonia;
using MyApp;

AppBuilder BuildAvaloniaApp()
   => AppBuilder.Configure<App>()
       .UsePlatformDetect()
       .WithInterFont()
       .LogToTrace();

var app = BuildAvaloniaApp();

app.StartWithClassicDesktopLifetime(args);



