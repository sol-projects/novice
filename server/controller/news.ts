import express, { Request, Response } from 'express';
import websites from '../scraper/websites';
import { INews, News } from '../model/News';
import * as Socket from '../socket/socket';
import { exec } from 'child_process';
import multer from 'multer';

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
    const updatedNews = await News.findByIdAndUpdate(id, updatedNewsData, {
      new: true,
      useFindAndModify: false,
    });

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
    console.log(`Evaluating website ${key} before pushing to database...`);
    const result = await value(req.body.n);
    for (let article of result) {
      let errorMessage = '';
      if (!isValidDate(article.date)) {
        errorMessage += `Invalid date (${article.date}): The article "${article.title}" on website ${key} should have a date in the current year. `;
      }

      if (!article.title || !article.content) {
        errorMessage += `Missing title or content: The article "${article.title}" on website ${key} should have a title and content. `;
      }

      if (
        article.authors.some(
          (author) => author.includes('/') || author.includes(',')
        )
      ) {
        errorMessage += `Invalid author format: The article "${article.title}" on website ${key} has an invalid author format. Authors should not contain "/" or ",". `;
      }

      if (hasDuplicates(article.categories)) {
        errorMessage += `Duplicate categories: The article "${article.title}" on website ${key} has duplicate categories. `;
      }

      if (hasDuplicates(article.authors)) {
        errorMessage += `Duplicate authors: The article "${article.title}" on website ${key} has duplicate authors. `;
      }

      if (!isValidURL(article.url)) {
        errorMessage += `Invalid URL: The article "${article.title}" on website ${key} has an invalid URL (${article.url}). `;
      }

      if (errorMessage) {
        console.error(
          `Invalid data found: ${errorMessage}Not pushing "${article.url}" to the database...`
        );
        continue;
      }

      const existingNews = await News.findOne({
        title: article.title,
        content: article.content,
      });

      if (!existingNews) {
        news.push(article);
      } else {
        console.log(
          `Article "${article.title}" on website ${key} already exists. Not pushing to database...`
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
    date: value.date
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

export async function addView(req: Request, res: Response) {
  try {
    const { id } = req.body;

    let date = new Date();
    await News.findByIdAndUpdate(id, {
      $push: { views: date },
    });

    res.status(201).json({ message: 'View created successfully' });
  } catch (error) {
    console.error(error);
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

    exec(
      './gradlew run --args="in.txt"',
      { cwd: '../geo-lang/interpreter' },
      function (error, stdout, stderr) {
        if (error) {
          console.error(error);
          return res
            .status(500)
            .send('Error executing the GeoLang interpreter');
        }

        fs.readFile(
          '../geo-lang/interpreter/app/out.geojson',
          'utf8',
          function (err: any, data: any) {
            if (err) {
              console.error(err);
              return res.status(500).send('Error reading the output file');
            }

            res.send(data);
          }
        );
      }
    );
  });
}

export async function findTextAreas(req: Request, res: Response) {
  const image = req.file;

  if (!image) {
    return res.status(400).send('No image file provided');
  }

  console.log(image.buffer);

  const fs = require('fs');
  fs.writeFile(
    '../newspaper_to_digital/input.png',
    image.buffer,
    'binary',
    function (err: any) {
      if (err) {
        console.error(err);
        return res.status(500).send('Error writing the image to input.png');
      }

      exec(
        'source env/bin/activate && python __init__.py --eval',
        { cwd: '../newspaper_to_digital' },
        function (error, stdout, stderr) {
          if (error) {
            console.error(error);
            return res
              .status(500)
              .send('Error during execution. Invalid file?');
          }

          fs.readFile(
            '../newspaper_to_digital/output.png',
            function (err: any, data: any) {
              if (err) {
                console.error(err);
                return res.status(500).send('Error reading the output file');
              }

              res.send(data);
            }
          );
        }
      );
    }
  );
}

export async function findImageSimilarity(req: Request, res: Response) {
  const files = req.files as Express.Multer.File[];

  if (!files || files.length !== 2) {
    return res.status(400).send('Two image files are required for comparison');
  }

  const fs = require('fs');
  const path = require('path');
  const tempImagePaths = files.map((file) => file.path);

 try {
    const files = req.files as Express.Multer.File[];
    if (!files || files.length !== 2) {
      return res.status(400).send('Two image files are required for comparison');
    }    const modelPath = path.resolve('../siamese_find_by_photo/checkpoints/siamese_network.pth');
    const torch = require('torch');
    const SiameseNetwork = require('./../siamese_find_by_photo/inference_siamese'); 
    const model = new SiameseNetwork();
    model.load_state_dict(torch.load(modelPath));
    model.eval();

    const getEmbedding = (imagePath: string) => {
      const { Image } = require('image-js');
      const img = Image.load(imagePath).resize({ width: 100, height: 100 });
      const imgTensor = torch.tensor(img.toArray()).unsqueeze(0);
      return model.forward_once(imgTensor);
    };

    const embeddings = tempImagePaths.map((imgPath) => getEmbedding(imgPath));
    const distance = torch.pairwise_distance(embeddings[0], embeddings[1]).item();

    tempImagePaths.forEach((filePath) => fs.unlinkSync(filePath));

    return res.json({ similarity: 1 - distance, distance });
  } catch (error) {
    console.error(error);

    tempImagePaths.forEach((filePath) => fs.unlinkSync(filePath));
    return res.status(500).send('Error processing the images');
  }
}

export async function findSportTypes(req: Request, res: Response) {
  const image = req.file;

  if (!image) {
    return res.status(400).send('No image file provided');
  }

  console.log(image.buffer);

  const fs = require('fs');
  fs.writeFile(
    '../sistem-za-razpoznavo-sport/input2.jpg',
    image.buffer,
    'binary',
    function (err: any) {
      if (err) {
        console.error(err);
        return res.status(500).send('Error writing the image to input.png');
      }

      exec(
        'python main.py',
        { cwd: '../sistem-za-razpoznavo-sport' },
        function (error, stdout, stderr) {
          if (error) {
            console.error(error);
            return res
              .status(500)
              .send('Error during execution. Invalid file?');
          }

          fs.readFile(
            '../sistem-za-razpoznavo-sport/outputh.txt',
            function (err: any, data: any) {
              if (err) {
                console.error(err);
                return res.status(500).send('Error reading the output file');
              }

              res.send(data);
            }
          );
        }
      );
    }
  );
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
