# Model card — next-minute Belgian system imbalance

## Doel

Voorspel de system imbalance in MW op `t+1 minuut` vanaf de laatste geaccepteerde Elia-observatie op `t`, plus de gecalibreerde kans op een toestandflip. Dit is niet bedoeld voor autonome trading, balancing-acties of veiligheidkritische dispatch.

## Labels

`target_next` is de geobserveerde system imbalance één minuut na de cutoff. De toestand gebruikt een hysteresis deadband (standaard ±10 MW). Een flip is alleen een overgang van bevestigde positief naar bevestigde negatief of omgekeerd; neutrale of onbekende toestanden maskeren de flip-loss.

## Invoer en architectuur

De feature engine gebruikt uitsluitend data die uiterlijk op de cutoff beschikbaar was: lokale imbalance/ACE/prijzen en afgeleiden, load/wind/solarcontext, bronleeftijd/masks, source-quality en kalenderfase. De architectuur is een compact drie-member ensemble met:

- causale dilated TCN voor lokale dynamiek;
- Transformer-encoder voor langzamer context;
- gated fusie met statische tijdkenmerken;
- Gaussian-mixture head voor point en p10/p90;
- flip classification head en hulpheads voor delta en langere horizons.

De members gebruiken seeds 17, 29 en 43. Inference is ONNX CPU-only. Isotonic calibratie wordt afzonderlijk gefit op een chronologische calibration slice.

## Training en promotie

Preprocessing past alleen op de trainperiode. Validatie, calibratie en test zijn chronologisch gescheiden met purge-gaps. Het bundle bewaart schemahash, checksums, modelversie, trainperiode, per-shard datasetdigest en evaluatiebestanden.

Een kandidaat moet MAE minstens 2% verbeteren tegenover persistence, betere flip PR-AUC halen dan de klassieke featurebewuste baseline, Brier met minstens 1% verbeteren, 75–85% intervaldekking halen en niet meer dan 5% regressie tonen in aanwezige critical cohorts. Cohorts omvatten toestand, volatiliteitsterciel, fase binnen het quarter-hour en source-quality.

## Beperkingen

- De actuele historische periode en werkelijke metrics zijn pas betrouwbaar na een volledige Colab-run en onberoerde testevaluatie; de meegeleverde testbundle is synthetisch en niet productief.
- Elia kan historische correcties publiceren; de pipeline bewaart versies en vermijdt toekomstinformatie, maar bronvertraging blijft een risico.
- MARI-marktregime en datasets kunnen structureel verschillen. Meng pre- en post-regime data niet zonder aparte evaluatie.
- Een hoge flipkans is een modeluitkomst, geen garantie en geen handelsadvies.
