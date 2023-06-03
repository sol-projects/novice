export async function geolang(code: string): Promise<any> {
  const response = await fetch('http://localhost:8000/news/geolang', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ code })
  });

  if (!response.ok) {
    console.error('Error executing geolang request');
  }

  const data = await response.json();
  return data;
}
