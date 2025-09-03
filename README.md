# haystack
My test project for Haystack


# To run
I am sure I will need more secrets in the .env but for now it suffices for 
the tests to run until the error is discovered. The OPENAI is not yet running 
so you can add any value there in the .env file as long as its there.

- Create an .env file with the following variables:
  - OPENAI_API_KEY=your_openai_key
  
- In the root directory run 
  ```console 
  python3.13 -m unittest tests/test_pipelines.py 