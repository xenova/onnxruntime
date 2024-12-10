using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;
public partial class AppiumSetup
{
    private static AppiumDriver? driver;

    public static AppiumDriver App => driver ?? throw new NullReferenceException("AppiumDriver is null");

    public AppiumSetup()
    {
        // If you started an Appium server manually, make sure to comment out the next line
        // This line starts a local Appium server for you as part of the test run
        AppiumServerHelper.StartAppiumLocalServer();

        var windowsOptions = new AppiumOptions
        {
            // Specify windows as the driver, typically don't need to change this
            AutomationName = "windows",
            // Always Windows for Windows
            PlatformName = "Windows",
            // The identifier of the deployed application to test
            App = "ORT.CSharp.Tests.MAUI_9zz4h110yvjzm!App",
        };

        // Note there are many more options that you can use to influence the app under test according to your needs
        //IntPtr hwnd = ((MauiWinUIWindow)Microsoft.Maui.Controls.Application.Current.Windows[0].Handler.PlatformView).WindowHandle;
        //windowsOptions.AddAdditionalAppiumOption("appTopLevelWindow", hwnd.ToString("x"));

        driver = new WindowsDriver(new Uri("http://127.0.0.1:4723/wd/hub"), windowsOptions);
    }

    public void Dispose()
    {
        driver?.Quit();

        // If an Appium server was started locally above, make sure we clean it up here
        AppiumServerHelper.DisposeAppiumLocalServer();
    }
}