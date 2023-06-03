import express, { Request, Response } from 'express';
import websites from '../scraper/websites';
import { INews, News } from '../model/News';
import * as Socket from '../socket/socket';
import { exec } from 'child_process';

async function run_query(res: Response, query: any) {
  try {
    const news = await News.find(query).exec();
    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to retrieve news from MongoDB');
  }
}

export async function id(req: Request, res: Response) {
  const { id } = req.params;

  try {
    const news = await News.findById(id);

    if (!news) {
      return res.status(404).json({ message: `News with id ${id} not found` });
    }

    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Server error');
  }
}

export async function remove(req: Request, res: Response) {
  const id = req.params.id;

  try {
    const deletedNews = await News.findByIdAndDelete(id);

    if (!deletedNews) {
      return res.status(404).send('News not found');
    }

    res.send(`News with ID ${id} deleted`);
  } catch (error) {
    console.error(error);
    res.status(500).send(`Failed to delete news with ID ${id}`);
  }
}

export async function update(req: Request, res: Response) {
  const id = req.params.id;
  const updatedNewsData = req.body;

  if (!id) {
    return res.status(400).send('ID is required for updating news.');
  }

  try {
    const updatedNews = await News.findByIdAndUpdate(
      id,
      updatedNewsData,
      { new: true, useFindAndModify: false }
    );

    if (!updatedNews) {
      return res.status(404).json({ message: `News with ID ${id} not found` });
    }

    res.json(updatedNews);

  } catch (error) {
    console.error(error);
    res.status(500).send('Server error while updating news');
  }
}

export async function all(req: Request, res: Response) {
  run_query(res, {});
}

export async function store(req: Request, res: Response) {
  let news: INews[] = [];
  for await (let [key, value] of websites) {
    const result = await value(req.body.n);
    console.log(`Evaluating website ${key} before pushing to database...`);
    for (let value of result) {
      let errorMessage = '';
      if (!isValidDate(value.date)) {
        errorMessage += `Invalid date (${value.date}): The article "${value.title}" on website ${key} should have a date in the current year. `;
      }

      if (!value.title || !value.content) {
        errorMessage += `Missing title or content: The article "${value.title}" on website ${key} should have a title and content. `;
      }

      if (value.authors.some(author => author.includes("/") || author.includes(","))) {
        errorMessage += `Invalid author format: The article "${value.title}" on website ${key} has an invalid author format. Authors should not contain "/" or ",". `;
      }

      if (hasDuplicates(value.categories)) {
        errorMessage += `Duplicate categories: The article "${value.title}" on website ${key} has duplicate categories. `;
      }

      if (hasDuplicates(value.authors)) {
        errorMessage += `Duplicate authors: The article "${value.title}" on website ${key} has duplicate authors. `;
      }

      if (!isValidURL(value.url)) {
        errorMessage += `Invalid URL: The article "${value.title}" on website ${key} has an invalid URL (${value.url}). `;
      }

      if (errorMessage) {
        console.error(`Invalid data found: ${errorMessage}Not pushing "${value.url}" to the database...`);
        continue;
      }

      const existingNews = await News.findOne({
        title: value.title,
        content: value.content,
      });

      if (!existingNews) {
        news.push(value);
      } else {
        console.log(
          `Article "${value.title}" on website ${key} already exists. Not pushing to database...`
        );
      }
    }

    console.log(`Website ${key} evaluated successfully...`);
  }

  news.sort((a, b) => a.date.getTime() - b.date.getTime());
  try {
    await News.create(news);
    Socket.emit('news-added', news);
    res.status(201).json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to save news to MongoDB');
  }
}

function isValidDate(date: Date) {
  const currentYear = new Date().getFullYear();
  const year = new Date(date).getFullYear();
  return year === currentYear;
}

function hasDuplicates(array: string[]) {
  return new Set(array).size !== array.length;
}

function isValidURL(url: string) {
  try {
    new URL(url);
    return true;
  } catch (error) {
    return false;
  }
}

export async function add(req: Request, res: Response) {
  let news: INews[] = [];

  let payload = req.body;

  if (Array.isArray(payload)) {
    payload = payload[0];
  }

  const value = payload;
  const existingNews = await News.findOne({
    title: value.title,
    content: value.content,
  });

  if (!existingNews) {
    news.push(value);
  } else {
    console.log(
      `Article "${value.title}" already exists. Not pushing to database...`
    );
  }

  console.log(`News ${value.title} evaluated successfully...`);

  try {
    await News.create(news);
    Socket.emit('news-added', news);
    res.status(201).json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to save news to MongoDB');
  }
}

export async function view(req: Request, res: Response) {
  try {
    const { id } = req.body;

    await News.findByIdAndUpdate(id, {
      $push: { views: { date: new Date() } },
    });

    res.status(201).json({ message: 'View created successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Error creating view', error });
  }
}

export async function scrape(req: Request, res: Response) {
  let news: INews[] = [];
  const website = await websites.get(req.params.website);

  if (!website) {
    res
      .status(500)
      .send(`Server error. Website with the name ${website} does not exist.`);
  } else {
    const result = await website(+req.params.n);

    for (let value of result) {
      news.push(value);
    }

    res.json(news);
  }
}

export async function scrapeAll(req: Request, res: Response) {
  let news: INews[] = [];

  for await (let [key, value] of websites) {
    const valueResult = await value(+req.params.n);
    for (let value of valueResult) {
      news.push(value);
    }
  }

  res.json(news);
}

export async function geolang(req: Request, res: Response) {
  const { code } = req.body;

  const fs = require('fs');
  fs.writeFile('../geo-lang/interpreter/app/in.txt', code, function (err: any) {
    if (err) {
      console.error(err);
      return res.status(500).send('Error writing the code to in.txt');
    }

    exec('gradle run --args="in.txt"', { cwd: '../geo-lang/interpreter' }, function (error, stdout, stderr) {
      if (error) {
        console.error(error);
        return res.status(500).send('Error executing the GeoLang interpreter');
      }

      fs.readFile('../geo-lang/interpreter/app/out.geojson', 'utf8', function (err: any, data: any) {
        if (err) {
          console.error(err);
          return res.status(500).send('Error reading the output file');
        }

        res.send(data);
      });
    });
  });
}

export namespace Filter {
  export async function categories(req: Request, res: Response) {
    run_query(res, { categories: { $all: req.params.categories.split(',') } });
  }

  export async function authors(req: Request, res: Response) {
    run_query(res, { authors: { $all: req.params.authors.split(',') } });
  }

  export function location(req: Request, res: Response) {
    run_query(res, { location: req.params.location });
  }

  export function website(req: Request, res: Response) {
    run_query(res, { url: { $regex: req.params.website, $options: 'i' } });
  }

  export function title(req: Request, res: Response) {
    run_query(res, { title: { $regex: req.params.title, $options: 'i' } });
  }

  export function content(req: Request, res: Response) {
    run_query(res, { content: { $regex: req.params.content, $options: 'i' } });
  }

  export namespace Date {
    export function before(req: Request, res: Response) {
      run_query(res, { date: { $lt: req.params.date } });
    }

    export function after(req: Request, res: Response) {
      run_query(res, { date: { $gt: req.params.date } });
    }

    export function range(req: Request, res: Response) {
      run_query(res, {
        date: {
          $gte: req.params.after,
          $lte: req.params.before,
        },
      });
    }
  }
}
