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
public class MainPageTests
{
    protected AppiumDriver App => AppiumSetup.App;

    [Fact]
    public void ClickRunAllTest()
    {
        var element = App.FindElement(By.XPath(".//*[text()='Run All  ►►']"));

        Assert.Equal(element.Text, "Run All  ►►");

        element.Click();

        String color = "SuccessfulTestsColor";
        String binding = "Passed";
        MobileElement label = driver.findElementByXPath("//Label[@TextColor='" + color + "' and @Text='" + binding + "']");
        String labelText = label.getText();
        System.out.println("The text displayed by the Label is: " + labelText);

        Task.Delay(600).Wait();
    }
}
