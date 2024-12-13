using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Interfaces;
using OpenQA.Selenium.Appium.Windows;
using OpenQA.Selenium.DevTools.V119.DOMStorage;
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

        IReadOnlyCollection<AppiumElement> btnElements = App.FindElements(By.XPath("//Button"));

        AppiumElement? btn = null;
        foreach (var element in btnElements)
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

        IReadOnlyCollection<AppiumElement> labelElements = App.FindElements(By.XPath("//Text"));
        int numPassed = -1;
        int numFailed = -1;

        for (int i = 0; i < labelElements.Count; i++)
        {
            AppiumElement element = labelElements.ElementAt(i);

            _output.WriteLine("We're at label element with tag name " + element.TagName);
            _output.WriteLine("We're at label element with text " + element.Text);
            
            if (element.Text.Equals("✔"))
            {
                _output.WriteLine("Matched with success indicator");
                i++;
                numPassed = int.Parse(labelElements.ElementAt(i).Text);
            }

            if (element.Text.Equals("⛔"))
            {
                _output.WriteLine("Matched with Failure indicator");
                i++;
                numFailed = int.Parse(labelElements.ElementAt(i).Text);
                element.Click();
                Task.Delay(1000).Wait();
                break;
            }
        }

        // if either of these are -1, then we couldn't find the correct labels;
        Assert.True(numPassed >= 0, "Could not find number passed label.");
        Assert.True(numFailed >= 0, "Could not find number failed label.");

        if (numFailed == 0)
        {
            // all tests passed! wahoo!
            return;
        }

        _output.WriteLine("on the results page =============================");

        IReadOnlyCollection<AppiumElement> elements = App.FindElements(By.XPath("//ComboBox"));

        foreach (var element in elements)
        {
            _output.WriteLine("We're at label element with tag name " + element.TagName);
            _output.WriteLine("We're at label element with text " + element.Text);

            if (element.Text.Equals("✔"))
            {
                _output.WriteLine("Matched with success indicator");
            }

            if (element.Text.Equals("⛔"))
            {
                _output.WriteLine("Matched with Failure indicator");
                //element.Click();
                //Task.Delay(5000).Wait();
                break;
            }
            element.Click();
            Task.Delay(500).Wait();
        }

        _output.WriteLine("clicked the combo box");

        IReadOnlyCollection<AppiumElement> elements2 = App.FindElements(By.XPath("//ListItem"));

        foreach (var element in elements2)
        {
            if (element.Text.Equals("Failed"))
            {
                element.Click();
                Task.Delay(500).Wait();
            }
        }

        StringBuilder sb = new StringBuilder();

        IReadOnlyCollection<AppiumElement> textResults = App.FindElements(By.XPath("//Text"));

        foreach (var element in textResults)
        {
            sb.AppendLine(element.Text);
        }

        Assert.True(numFailed == 0, sb.ToString());

    }
}
