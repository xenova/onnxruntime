using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
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

        Task.Delay(600).Wait();
    }
}
