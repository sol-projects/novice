import cheerio from 'cheerio'
import axios from 'axios'

async function _gov(n: number) {
    axios.get('https://www.gov.si/novice/')
          .then(response => {
            const $ = cheerio.load(response.data);
            const titles: string[] = [];
            const urls: string[] = [];

            $('.title').each((i, element) => {
              const title: string = $(element).text();
              const url: string | undefined = $(element).find('a').attr('href');
              if (url) {
                urls.push("https://www.gov.si" + url);
              }
              titles.push(title.trim());
                if(i == n - 1) {
                    return false;
                }
            });

            console.log('Titles:', titles);
            console.log('URLs:', urls);
          })
          .catch(error => {
            console.log(error);
          });

        return ""

}

export = _gov;
