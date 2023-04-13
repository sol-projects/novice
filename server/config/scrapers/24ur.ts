import cheerio from 'cheerio'
import axios from 'axios'

async function _24ur() {
    axios.get('https://www.24ur.com/novice')
          .then(response => {
            const $ = cheerio.load(response.data);
            const titles: string[] = [];
            const urls: string[] = [];

            $('a:not(:has(div:contains("OGLAS")))').each((i, element) => {
              const href = $(element).attr('href');
              if (href && href.startsWith('/novice/')) {
                  let title = $(element).find('div h1, div h2, div h3, div h4, div h5, div h6')
                  if(title.text().trim().length > 0) {
                urls.push('https://www.24ur.com' + href);
                titles.push(title.text().trim());
                  } else {
                      let title2 = $(element).find('h1 span, h2 span, h3 span, h4 span, h5 span, h6 span').text().trim()
                    if(title2.length > 0) {
                        urls.push('https://www.24ur.com' + href);
                        titles.push(title2);
                    }
                  }
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

export = _24ur;
