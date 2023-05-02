import cheerio from 'cheerio';
import axios from 'axios';

async function _delo(n: number) {
  axios
    .get('https://www.delo.si/zadnje/')
    .then((response) => {
      const $ = cheerio.load(response.data);
      const titles: string[] = [];
      const urls: string[] = [];

      $('.paginator_item').each((i, element) => {
        const title: string = $(element)
          .find('.article_teaser_timeline__title_text')
          .text();
        const url: string | undefined = $(element).find('a').attr('href');
        if (url) {
          urls.push('https://www.delo.si' + url);
        }
        titles.push(title.trim());

        if (i == n - 1) {
          return false;
        }
      });

      console.log('Titles:', titles);
      console.log('URLs:', urls);
    })
    .catch((error) => {
      console.log(error);
    });

  return '';
}

export = _delo;
