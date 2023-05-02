import cheerio from 'cheerio';
import axios from 'axios';

async function _siol(n: number) {
  const date = new Date();
  axios
    .get(
      `https://siol.net/pregled-dneva/${date.getFullYear()}-${
        date.getMonth() + 1
      }-${date.getDate()}`
    )
    .then((response) => {
      const $ = cheerio.load(response.data);
      const titles: string[] = [];
      const urls: string[] = [];

      $('.timemachine__article_item').each((i, element) => {
        const title: string = $(element).find('.card__title').text();
        const url: string | undefined = $(element).find('a').attr('href');
        if (url) {
          urls.push('https://siol.net' + url);
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

export = _siol;
