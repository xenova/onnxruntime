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
    public async void ClickRunAllTest()
    {
        Console.WriteLine("In the ClickRunAllTest");

        //var element = App.FindElement(By.XPath("//Button[contains(@Text, 'Run All')]"));

        //Console.WriteLine("found element with text", element.Text);

        //Assert.Equal("Run All  ►►", element.Text);

        //element.Click();

        //String color = "SuccessfulTestsColor";
        //String binding = "Passed";
        //MobileElement label = driver.findElementByXPath("//Label[@TextColor='" + color + "' and @Text='" + binding + "']");
        //String labelText = label.getText();
        //System.out.println("The text displayed by the Label is: " + labelText);

        //await Task.Delay(600).Wait();

        IReadOnlyCollection<AppiumElement> elements = App.FindElements(By.XPath("//Button"));

        AppiumElement btn;
        foreach (var element in elements)
        {
            if (element.Text.Contains("Run All"))
            {
                btn = element;
                element.Click();
                break;
            }
        }

        


        

    }
}
