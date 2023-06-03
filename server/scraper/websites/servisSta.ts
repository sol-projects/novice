import { INews } from '../../model/News';
import cheerio from 'cheerio';
import axios from 'axios';
import { parse } from 'date-fns';
import * as Db from '../../db/db';

function extractCountryFromTitle(title: string): string {
  const countryPrefixes = ['v ', 'v ', 'na '];

  for (const prefix of countryPrefixes) {
    const startIndex = title.indexOf(prefix);
    if (startIndex !== -1) {
      const endIndex = title.indexOf(' ', startIndex + prefix.length);
      if (endIndex !== -1) {
        return title.substring(startIndex + prefix.length, endIndex);
      }
    }
  }

  return '';
}

async function servisSta(numArticlesToOpen: number): Promise<INews[]> {
  const response = await axios.get('https://servis.sta.si/');
  const $ = cheerio.load(response.data);

  const newsList: INews[] = [];

  const articleElements = $('article.item');

  for (let i = 0; i < numArticlesToOpen && i < articleElements.length; i++) {
    const articleElement = articleElements.eq(i);

    const url = articleElement.find('a').attr('href');

    if (!url) {
      continue; // Skip this iteration if url is undefined
    }

    const articleResponse = await axios.get(`https://servis.sta.si${url}`);
    const article$ = cheerio.load(articleResponse.data);

    const title = article$('article.articleui h1')
      .text()
      .replace(/\n/g, '')
      .trim();

    const leadElement = article$('article').find('.lead');
    const lead = leadElement.text() || '';

    const textElements = article$('article').find('.text');
    const preTextElement = textElements.eq(0).find('pre');
    const preText = preTextElement.length > 0 ? preTextElement.text() : '';

    const contentTmp = lead || preText ? `${lead} ${preText}` : '';
    const content = contentTmp.replace(/\n/g, '').trim();

    const categoryElement = article$('aside.articlemeta').find(
      'div.items > div:nth-child(2)'
    );
    const categories = [
      categoryElement.text().replace('Kategorija:', '').trim().toLowerCase(),
    ];

    const authorElement = article$('aside.articlemeta').find(
      'div.items > div:nth-child(4)'
    );
    const authors = authorElement
      .text()
      .replace('Avtor:', '')
      .trim()
      .split('/');

    let location = extractCountryFromTitle(title);
    if(location == '') {
        location = Db.Util.getFirstSettlement(categories);
    }

    const coords = await Db.Util.toCoords(location);

    const date = parse('2000-10-10', 'yyyy-MM-dd', new Date());

    const news: INews = {
      title,
      url: `https://servis.sta.si${url}`,
      date,
      authors,
      content,
      categories,
      views: [],
      location: {
        type: 'Point',
        coordinates: coords,
      },
    };

    newsList.push(news);

    // Wait or introduce delay if necessary before making the next request
  }

  return newsList;
}

export = servisSta;
