using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace XssedCrawler {
	/// <summary>
	/// Class to handle any saving to the file system
	/// </summary>
	public static class FileManager {

		public const string URL_LIST_FILE = @"Url List.txt";
		public const string VULN_URL_LIST_FILE = @"Vulnerable URLs.txt";
		public const string ARFF_FILE = @"classified_urls.arff";
		public const string JS_EVENT_LIST_FILE = @"javascript_events.txt";
		public const string DOM_EVENT_LIST_FILE = "dom_events.txt";

		/// <summary>
		/// Saves an html page as Webpage_"pageNumber"
		/// </summary>
		/// <param name="htmlData">A string containing the entire html page</param>
		/// <param name="pageNumber">The number of the page in the crawl</param>
		public static void SaveHtmlToDisk(string htmlData, int pageNumber) {
			string fileName = String.Format("Webpage_{0}", pageNumber);
			string filePath = String.Format(@"webpage\{0}.txt", fileName); //note this will save in Debug or Release folder if run in VS
			File.WriteAllText(filePath, htmlData);
		}

		/// <summary>
		/// Saves a list of urls to a single txt document
		/// </summary>
		public static void SaveUrlListToDisk(IEnumerable<String> urls) {
			File.AppendAllLines(URL_LIST_FILE, urls.ToArray());
		}

        public static void SaveVulnerableUrlListToDisk(IEnumerable<String> urls)
        {
            File.AppendAllLines(VULN_URL_LIST_FILE, urls.ToArray());
        }

		/// <summary>
		/// Load the list of javascript handlers from a file
		/// </summary>
		public static IEnumerable<string> LoadJavascriptEvents() {
			IEnumerable<string> events = File.ReadLines(JS_EVENT_LIST_FILE);
			return events;
		}

		/// <summary>
		/// Load the list of DOM handlers from a file
		/// </summary>
		public static IEnumerable<string> LoadDomEvents() {
			IEnumerable<string> events = File.ReadLines(DOM_EVENT_LIST_FILE);
			return events;
		}

		/// <summary>
		/// Read in both URL lists, classify the contents, and save the results to a new file.
		/// </summary>
		public static void ClassifyAll()
		{
			var c = new List<Classification>();
			
			c.AddRange(File.ReadLines(URL_LIST_FILE).Select(negativeUrl => new Classification(negativeUrl, "FALSE")));
			c.AddRange(File.ReadLines(VULN_URL_LIST_FILE).Select(positiveUrl => new Classification(positiveUrl, "TRUE")));
			writeClassifiedToArff(c);
		}

		private static void writeArffBody(List<Classification> classifiedUrls, StreamWriter s) {
			foreach (var c in classifiedUrls) {
				s.WriteLine("{0}, {1}, {2}, {3}, {4}, {5}", c.Characters, c.ScriptCount, c.EncodedCharacters, c.JsEventHandlers, c.DomEvents, c.Class);
			}
		}

		private static void writeArffHeader(StreamWriter s) {
			s.WriteLine("@RELATION xss\n");
			foreach (var property in typeof(Classification).GetFields()) {
				if (property.Name == "Class") s.WriteLine("@ATTRIBUTE {0} {{TRUE, FALSE}}", property.Name);
				else s.WriteLine("@ATTRIBUTE {0} NUMERIC", property.Name);
			}
			s.WriteLine("\n@DATA");
		}

		private static void writeClassifiedToArff(List<Classification> classifiedUrls) {
			File.Create(ARFF_FILE).Close();
			StreamWriter s = File.AppendText(ARFF_FILE);
			writeArffHeader(s);
			writeArffBody(classifiedUrls, s);
			s.Close();
		}
	}
}
