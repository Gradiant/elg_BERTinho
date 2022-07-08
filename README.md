# ``Bertinho`` monolingual BERT model for Galician

A pre-trained BERT model for Galician (12layers, cased). Trained on Wikipedia.

This repository contains a dockerized API built over Bertinho for integrate it into the ELG. Bertinho is a pre-trained BERT model for Galician (12layers, cased) trained on Wikipedia.

Its original code can be found [here](https://huggingface.co/dvilares/bertinho-gl-base-cased).

# Usage


## Install

```
source docker-build.sh
```

## Run
```
docker run --rm -p 0.0.0.0:8866:8866 --name bertinho elg_bertinho:1.0.1
```

## Use

```
    curl -X POST http://0.0.0.0:8866/predict_bertinho -H 'Content-Type: application/json' -d '{"type": "text","content":"Onte fumos buscar unha nova freidora ao [MASK] do pobo."}'
```


Result:

```
[{"content":"Onte fumos buscar unha nova freidora ao servizo do pobo.","score":0.7600728869438171},
{"content":"Onte fumos buscar unha nova freidora ao redor do pobo.","score":0.041864052414894104},
{"content":"Onte fumos buscar unha nova freidora ao p\u00e9 do pobo.","score":0.022877756506204605},
{"content":"Onte fumos buscar unha nova freidora ao car\u00f3n do pobo.","score":0.02281048335134983},
{"content":"Onte fumos buscar unha nova freidora ao alcance do pobo.","score":0.02227281592786312}]

```

## Test
In the folder `test` you have the files for testing the API according to the ELG specifications.
It uses an API that acts as a proxy with your dockerized API that checks both the requests and the responses.
For this follow the instructions:

1) Launch the test: `sudo docker-compose --env-file bertinho.env up`

2) Make the requests, instead of to your API's endpoint, to the test's endpoint:
   ```
      curl -X POST  http://0.0.0.0:8866/processText/service -H 'Content-Type: application/json' -d '{"type": "text", "content":"Exemplo de [MASK] como entrada ao programa."}'
   ```
3) If your request and the API's response is compliance with the ELG API, you will receive the response.
   1) If the request is incorrect: Probably you will don't have a response and the test tool will not show any message in logs.
   2) If the response is incorrect: You will see in the logs that the request is proxied to your API, that it answers, but the test tool does not accept that response. You must analyze the logs.

## Citation
The original work of this tool is:
- https://huggingface.co/dvilares/bertinho-gl-base-cased
- More info: https://arxiv.org/abs/2103.13799v1

