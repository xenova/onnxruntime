using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Interfaces;
using OpenQA.Selenium.Appium.Windows;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests;
// Add a CollectionDefinition together with a ICollectionFixture
// to ensure that setting up the Appium server only runs once
// xUnit does not have a built-in concept of a fixture that only runs once for the whole test set.
[CollectionDefinition("UITests")]
public sealed class UITestsCollectionDefinition : ICollectionFixture<AppiumSetup>
{

}

// Add all tests to the same collection as above so that the Appium server is only setup once
[Collection("UITests")]
public class MainPageTests
{
    protected AppiumDriver App => AppiumSetup.App;

    [Fact]
    public void ClickRunAllTest()
    {
        var element = App.FindElement(By.XPath(".//*[text()='Run All  ►►']"));

        Assert.Equal(element.Text, "Run All  ►►");

        element.Click();

        //String color = "SuccessfulTestsColor";
        //String binding = "Passed";
        //MobileElement label = driver.findElementByXPath("//Label[@TextColor='" + color + "' and @Text='" + binding + "']");
        //String labelText = label.getText();
        //System.out.println("The text displayed by the Label is: " + labelText);

        Task.Delay(600).Wait();
    }
}
