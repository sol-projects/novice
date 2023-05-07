import mongoose from 'mongoose';

export function connect() {
  if (!process.env) {
    console.error(
      'You must call dotenv.config() before calling this function.'
    );
  } else {
    mongoose.connect(
      `mongodb+srv://${process.env.DB_USERNAME}:${process.env.DB_PASSWORD}@cluster0.bd0tfwp.mongodb.net/?retryWrites=true&w=majority`
    );
  }
}

export namespace Util {
  export async function toCoords(place: string): Promise<[number, number]> {
    const url = `https://geokeo.com/geocode/v1/search.php?q=${place}&api=${process.env.GEOKEO_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    const results = data.results;
    const slovenianCities = results.filter((result: any) => {
      return (
        result.address_components.country === 'Slovenia' &&
        result.class === 'place' &&
        result.type === 'city'
      );
    });

    if (slovenianCities.length === 0) {
      return [
        Number(results[0].geometry.location.lng),
        Number(results[0].geometry.location.lat),
      ];
    }

    return [
      Number(slovenianCities[0].geometry.location.lng),
      Number(slovenianCities[0].geometry.location.lat),
    ];
  }
}
