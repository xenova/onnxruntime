using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Interfaces;
using OpenQA.Selenium.Appium.Windows;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI.UITest;
// Add a CollectionDefinition together with a ICollectionFixture
// to ensure that setting up the Appium server only runs once
// xUnit does not have a built-in concept of a fixture that only runs once for the whole test set.
[CollectionDefinition("Microsoft.ML.OnnxRuntime.Tests.MAUI")]
public sealed class UITestsCollectionDefinition : ICollectionFixture<AppiumSetup>
{

}

// Add all tests to the same collection as above so that the Appium server is only setup once
[Collection("Microsoft.ML.OnnxRuntime.Tests.MAUI")]
public class MainPageTests
{
    protected AppiumDriver App => AppiumSetup.App;

    private readonly ITestOutputHelper _output;

    public MainPageTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void FailingOutputTest()
    {
        Console.WriteLine("This is test output.");
        throw new Exception("This is meant to fail.");
    }

    [Fact]
    public void SuccessfulTest()
    {
        Assert.True(true);
    }


    [Fact]
    public async Task ClickRunAllTest()
    {
        Console.WriteLine("In the ClickRunAllTest");

        IReadOnlyCollection<AppiumElement> elements = App.FindElements(By.XPath("//Button"));

        AppiumElement? btn = null;
        foreach (var element in elements)
        {
            _output.WriteLine("We're at element " + element.Text);
            if (element.Text.Contains("Run All"))
            {
                _output.WriteLine("Found run all button");
                _output.WriteLine("");
                btn = element;
                element.Click();
                break;
            }
        }

        Assert.NotNull(btn ?? throw new Xunit.Sdk.XunitException("Run All button was not found."));

        while (!btn.Enabled)
        {
            // whille the button is disabled, then wait half a second
            Task.Delay(500).Wait();
        }

        _output.WriteLine("BUTTON IS ENABLED AGAIN");

    }
}
