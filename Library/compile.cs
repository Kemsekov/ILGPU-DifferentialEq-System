using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using System.Reflection;
using System.Text;

public class DynamicCompilation
{
    //install Microsoft.CodeAnalysis
    ///dynamically compiles C# function from string and returns delegate to it
    /// csCode - code that contains TReturn Execute(TParam){...} function
    /// referencesAssembly - list of types references to which you may need
    public static Func<TParam, TReturn> CompileFunction<TParam, TReturn>(string csCode, params System.Type[] referencesAssembly)
    where TParam : notnull
    {
        var codeWithClass = "public class DynamicallyCompiled{ public static " + csCode + "}";
        var syntaxTree = CSharpSyntaxTree.ParseText(codeWithClass);
        // Set the compilation options
        var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
        .WithAllowUnsafe(true)
        .WithOptimizationLevel(OptimizationLevel.Release);
        // Add references
        var dd = typeof(Enumerable).GetTypeInfo().Assembly.Location;
        var coreDir = Directory.GetParent(dd) ?? throw new Exception();
        var runtimePath = coreDir.FullName + Path.DirectorySeparatorChar;
        var references = new List<MetadataReference>()
    {
        MetadataReference.CreateFromFile(typeof(TParam).Assembly.Location),
        MetadataReference.CreateFromFile(typeof(TReturn).Assembly.Location),
        //some basic references
        MetadataReference.CreateFromFile(Assembly.Load("netstandard").Location),
        MetadataReference.CreateFromFile(runtimePath + "mscorlib.dll"),
        MetadataReference.CreateFromFile(runtimePath + "System.Runtime.dll"),
    };
        references.AddRange(referencesAssembly.Select(i => MetadataReference.CreateFromFile(i.Assembly.Location)));
        // Create the compilation
        var compilation = CSharpCompilation.Create("DynamicAssembly", new[] { syntaxTree }, references, options);

        // Generate the assembly in memory
        using var stream = new MemoryStream();
        // Emit the IL code
        EmitResult result = compilation.Emit(stream);

        // Check for errors
        if (!result.Success)
        {
            var errorMessage = new StringBuilder();
            // Display the errors
            foreach (Diagnostic error in result.Diagnostics)
            {
                errorMessage.Append(error.GetMessage() + "\n");
            }
            throw new ArgumentException(errorMessage.ToString());
        }
        // Load the assembly
        stream.Seek(0, SeekOrigin.Begin);
        var assembly = Assembly.Load(stream.ToArray());

        // Get the type
        var type = assembly.GetType("DynamicallyCompiled");
        if (type is null)
        {
            throw new ArgumentException("DynamicallyCompiled class is not found!");
        }

        // Get the method
        var method = type.GetMethod("Execute");

        if (method is null)
        {
            throw new ArgumentException("DynamicallyCompiled.Execute method is not found!");
        }
        // Create a delegate
        var func = (Func<TParam, TReturn>)Delegate.CreateDelegate(typeof(Func<TParam, TReturn>), method);
        return func;
    }
}
