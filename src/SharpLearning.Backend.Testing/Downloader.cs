using System.IO;
using System.Net;

namespace SharpLearning.Backend.Testing
{
    public static class Downloader
    {
        public static Stream MaybeDownload(string urlBase, string fileName, string destinationDirectory)
        {
            if (!Directory.Exists(destinationDirectory))
            { Directory.CreateDirectory(destinationDirectory); }

            var target = Path.Combine(destinationDirectory, fileName);
            if (!File.Exists(target))
            {
                var wc = new WebClient();
                wc.DownloadFile(urlBase + fileName, target);
            }
            return File.OpenRead(target);
        }
    }
}
