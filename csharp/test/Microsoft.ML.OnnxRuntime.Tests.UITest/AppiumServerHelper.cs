using OpenQA.Selenium.Appium.Service;
using OpenQA.Selenium.Appium.Service.Options;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI.UITest;
public static class AppiumServerHelper
{
    private static AppiumLocalService? _appiumLocalService;

    public const string DefaultHostAddress = "127.0.0.1";
    public const int DefaultHostPort = 4723;

    public static void StartAppiumLocalServer(string host = DefaultHostAddress,
        int port = DefaultHostPort)
    {
        if (_appiumLocalService is not null)
        {
            return;
        }

        // this is needed on windows
        var options = new OptionCollector();
        options.AddArguments(new KeyValuePair<string, string>("--base-path", "/wd/hub"));
        options.AddArguments(new KeyValuePair<string, string>("--log-level", "debug"));

        var builder = new AppiumServiceBuilder()
            .WithIPAddress(host)
            .WithArguments(options)
            .WithLogFile(new FileInfo(@"C:\Users\carolinezhu\Documents\cowboy-coding\appium-server-logs\server_for_ort_separate_proj.log"))
            .UsingPort(port);

        // Start the server with the builder
        _appiumLocalService = builder.Build();
        _appiumLocalService.Start();
    }

    public static void DisposeAppiumLocalServer()
    {
        _appiumLocalService?.Dispose();
    }
}
