import puppeteer from 'puppeteer';
import { INews } from '../../model/News';

async function _mbinfo(n: number = 5) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://mariborinfo.com/lokalno');

  const links = await page.$$('a[href^="/novica/"]');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];
    const titleElement = await page.$('span.title__value');
    const titleText = await (titleElement
      ? titleElement.evaluate((e) => (e as HTMLElement).innerText)
      : '');
    //if (!title) continue;
    const url = `https://mariborinfo.com${await link.evaluate((e) =>
      e.getAttribute('href')
    )}`;

    const articlePage = await browser.newPage();
    await articlePage.goto(url);
    //await articlePage.waitForSelector('div.field field--name-field-besedilo', { visible: true });
    const authors = await articlePage.$$eval('.page-title-meta.block.block--app-breadcrumb.block--app .username__name', (els) =>
      els.map((e) => (e as HTMLElement).innerText.split('/')[0].trim())
    );

    const title = titleText.trim();

    const date = new Date();
    const categoryLinks = await articlePage.$$('a[href*="/tags/"]');
    const categories = await Promise.all(
      categoryLinks.map((link) =>
        link.evaluate((e) => (e as HTMLElement).innerText.trim().toLowerCase())
      )
    );

    const contentElements = await articlePage.$$eval(
      'div.field.field--name-field-besedilo',
      (elements) => elements.map((e) => (e as HTMLElement).innerText.trim())
    );

    const content = contentElements.join('\n');

    news.push({
      title,
      url,
      date: new Date(),
      authors,
      content,
      categories,
      views: [],
      location: {
        type: 'Point',
        coordinates: [0, 0],
      },
    });

    await articlePage.close();
  }

  await browser.close();

  return news;
}

export = _mbinfo;
