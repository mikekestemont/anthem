# Who Wrote the Wilhelmus?

## This repository
This repository holds the original code and data which was used for the following publication:

> Kestemont, M.; Stronks, E.; De Bruin, M. & De Winkel, T. *Van wie is het Wilhelmus? Auteurskwesties rond het Nederlandse volkslied met de computer onderzocht*. Amsterdam: Amsterdam University Press (2017).

![Cover image book Datheen](https://user-images.githubusercontent.com/4376879/32371233-69478378-c090-11e7-921d-b534ad03d532.png)

In this book, we use computational stylometry (i.e. quantitative stylistics via machine learning) to argue that Petrus Datheen (ca. 1531 - 1588), for various reasons, has been an overlooked candidate in the historical search for the author behind the 16th century Dutch-language national anthem of The Netherlands - the so-called *Wilhelmus*, which survives anonymously.

## Code
The code requires Python 3.4+ and depends on a series of common external packages from the scientific Python ecosystem, which are mostly part of the [Anaconda distribution](https://anaconda.org/), which we warmly recommend. The code in this repository and has only been tested on Unix-like platforms and is merely provided on a "as-is" basis for the purpose of replication purposes. Although we are unable to offer any long-term support, we are happy to try and clarify small issues about our implementation via GitHub issues in this repository.

## Data
The data in this repository has exclusively been harvested from the [Digitale Bibliotheek voor de Nederlandse letteren](http://dbnl.org/) and the (Liederenbank)[http://www.liederenbank.nl/]. Because these datasets are open to the general public online, we believe that this gives us the right to redistribute this data here in an enriched format that might inspire re-use. The data has been tagged and lemmatized using a [Memory-Based Tagger](https://arxiv.org/pdf/cmp-lg/9607012.pdf), which was trained on a Middle Dutch corpus. Although we are not allowed to redistribute the original training corpus here, we do include some additional training data which was manually corrected in the context of this project (under `data/tagged/corrected/`).

## Citation
If you use these materials in academic research, please consider citing the following publications:
- Kestemont, M.; Stronks, E.; De Bruin, M. & De Winkel, T. *Van wie is het Wilhelmus? Auteurskwesties rond het Nederlandse volkslied met de computer onderzocht*. Amsterdam: Amsterdam University Press (2017) [for the case study].
- Kestemont, M., Stover, J., Koppel, M., Karsdorp, K. & Daelemans, W., ‘Authenticating the writings of Julius Caesar’. In: *Expert Systems with Applications* 63 (2016): pp. 86–96 [for the verification method].
