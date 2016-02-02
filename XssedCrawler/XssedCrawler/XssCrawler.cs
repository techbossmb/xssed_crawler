using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

///modified for data incubators challenge
namespace XssedCrawler {
	public class XssCrawler {
		private const string BASE_URL = @"http://www.xssed.com";
		private const string ARCHIVE_URL = BASE_URL + @"/archive/page=";
		private int xssPage = 1;
		private List<String> urls = new List<String>(46000); //a little less than 46000 total archived pages
		
		/// <summary>
		/// Crawl through xssed.com to save the vulnerable url and the archived html of the vulnerable page.
		/// </summary>
		/// <remarks>
		/// This takes a long time ~7hours. Not all archived pages have any useful html.
		/// </remarks>
		public void Crawl() {
			int maxPage = 1530; //the website hasn't been updated in a while, so this is valid now. might need to dynamically set this if the website becomes active
			for (int page = 1; page <= maxPage; page++) {
				string data = tryGetUrlData(ARCHIVE_URL + page);
				var mirrors = WebPage.ParseWebPage(data, @"/mirror/\w*");
				getVulnerablePages(mirrors);
                Console.CancelKeyPress += cancelEventHandler;
			}
			FileManager.SaveUrlListToDisk(urls);
		}

        private void cancelEventHandler(object sender, ConsoleCancelEventArgs e)
        {
            FileManager.SaveVulnerableUrlListToDisk(urls);
            
        }
		private void getVulnerablePages(MatchCollection mirrors) {
			foreach (var pageUrl in mirrors) {
				string data = tryGetUrlData(BASE_URL + pageUrl);
				extractUrl(data);
				extractAndSaveHtmlPage(data);
			}
		}

		private void extractAndSaveHtmlPage(string data) {
			var matches = WebPage.ParseWebPage(data, "http://vuln.xssed.net(.*?)(?=\\\">)");
			string url = matches[0].ToString();
			data = tryGetUrlData(url);
			FileManager.SaveHtmlToDisk(data, xssPage++);
		}

		private void extractUrl(string data) {
			var matches = WebPage.ParseWebPage(data, @"(?<=URL: )https?:\/\/(.*?)(?=<\/t)");
			string url = matches[0].ToString();
			url = String.Join("", (url.Split(new[] { "<br>" }, StringSplitOptions.None))); //strip out <br> tags from the URL
			urls.Add(url);
		}

		private string tryGetUrlData(string url) {
			string data = "";
			try {
				data = getUrlData(url);
			}
			catch (Exception) { 
			} //lots of possible exceptions that would get in the way of the crawl - some URL null, some invalid, some 404, etc.
			return data;
		}

		private string getUrlData(string url) {
			StreamReader reader = WebPage.GetData(url);
			return reader.ReadToEnd();
		}
	}
}
