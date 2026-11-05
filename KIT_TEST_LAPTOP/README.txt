==============================================================================
  KIT USB - VERIFICA RAPIDA LAPTOP GAMING USATO (10-15 minuti)
  Modello di riferimento: ASUS TUF Gaming A15 2023 - FA507NV
  Ryzen 7 7735HS | RTX 4060 Laptop 8 GB | 16 GB DDR5 | SSD NVMe 512 GB
==============================================================================

INDICE
  1. Cosa c'è nella chiavetta
  2. PREPARAZIONE A CASA (una volta sola, serve Internet)
  3. Sul posto: cosa dire al venditore
  4. PROCEDURA CRONOMETRATA (15 minuti)
  5. GPU-Z: valori attesi
  6. HWiNFO: valori da osservare
  7. Temperature: normali o preoccupanti?
  8. CHECKLIST FINALE: comprare o no?
  9. Controlli extra (se il venditore concede 3-5 minuti in più)
 10. Problemi comuni e soluzioni
 11. Versioni dei programmi e fonti ufficiali

Una versione da stampare (1 foglio) è in: Documenti\SCHEDA_DA_STAMPARE.txt


------------------------------------------------------------------------------
1. COSA C'È NELLA CHIAVETTA
------------------------------------------------------------------------------

  KIT_TEST_LAPTOP\
  |-- README.txt                     <- questo file
  |-- 0_PREPARA_CHIAVETTA_A_CASA.bat <- scarica i programmi (A CASA, con Internet)
  |-- 1_CONTROLLO_AUTOMATICO.bat     <- controllo automatico di sistema (1 min)
  |-- 2_MONITOR_GPU_8_MINUTI.bat     <- registra e giudica la GPU durante OCCT
  |-- 3_ANALIZZA_LOG_HWINFO.bat      <- riassume il log di HWiNFO (CPU, GPU, SSD)
  |-- Programmi\
  |   |-- GPU-Z\GPU-Z.exe            <- identità e caratteristiche della GPU
  |   |-- HWiNFO64\HWiNFO64.exe      <- sensori: temperature, consumi, frequenze
  |   `-- OCCT\OCCT.exe              <- test di stabilità della GPU
  |-- Documenti\SCHEDA_DA_STAMPARE.txt
  |-- Extra\test_schermo.html        <- pixel morti / uniformità schermo
  |-- Extra\test_tastiera.html       <- test di tutti i tasti
  |-- Rapporti\                      <- qui vengono salvati tutti i risultati
  `-- Script\                        <- script PowerShell usati dai .bat

Gli script:
  - sono in SOLA LETTURA: non installano nulla, non modificano impostazioni,
    non mandano dati su Internet;
  - funzionano SENZA Internet;
  - sono file di testo: puoi aprirli con il Blocco note e leggerli.
  - Sono un AIUTO: se per qualche motivo non partono, tutta la procedura si
    può fare a mano seguendo questo README.

Programmi (tutti gratuiti, nessun account, nessuna installazione):
  - GPU-Z (TechPowerUp)        - all'avvio chiede se installarlo: rispondi "No".
  - HWiNFO64 Portable (REALiX) - versione portatile ufficiale (ZIP).
  - OCCT Personal (OCBASE)     - gratuito per uso personale, file .exe unico.
Tutti compatibili con Windows 10 e Windows 11 a 64 bit.


------------------------------------------------------------------------------
2. PREPARAZIONE A CASA (una volta sola, circa 5 minuti, serve Internet)
------------------------------------------------------------------------------

  1. Chiavetta USB da almeno 1 GB (FAT32, exFAT o NTFS vanno bene).
  2. Copia l'intera cartella KIT_TEST_LAPTOP nella chiavetta.
     (Se l'hai scaricata come ZIP: tasto destro sullo ZIP > Proprietà >
     spunta "Annulla blocco" > OK, POI estrai. Evita avvisi di Windows.)
  3. Dalla chiavetta, doppio clic su 0_PREPARA_CHIAVETTA_A_CASA.bat
     Lo script:
       - prova a scaricare HWiNFO e OCCT direttamente dai siti ufficiali;
       - per GPU-Z (e per gli altri se il download automatico non riesce)
         APRE LA PAGINA UFFICIALE nel browser: scarichi il file normalmente,
         torni alla finestra nera e premi INVIO. Lo script trova il file
         nella cartella Download e lo mette al posto giusto;
       - controlla la FIRMA DIGITALE di ogni programma (deve risultare OK);
       - scrive versioni e impronte SHA256 in Rapporti\VERSIONI_PROGRAMMI.txt.
  4. FORTEMENTE CONSIGLIATO: prova tutto una volta sul tuo PC (anche senza
     GPU NVIDIA): apri GPU-Z, HWiNFO, OCCT e lancia 1_CONTROLLO_AUTOMATICO.
     Sul posto sarai molto più veloce e tranquillo.
  5. Opzionale: stampa Documenti\SCHEDA_DA_STAMPARE.txt.

  Download manuale (alternativa allo script), SOLO da questi indirizzi:
    GPU-Z  : https://www.techpowerup.com/download/techpowerup-gpu-z/
             versione standard (non "ASUS ROG") -> Programmi\GPU-Z\GPU-Z.exe
    HWiNFO : https://www.hwinfo.com/download/
             "Portable" (ZIP) -> estrai TUTTO in Programmi\HWiNFO64\
             poi copia Script\HWiNFO64.INI.modello in Programmi\HWiNFO64\
             rinominandolo HWiNFO64.INI
    OCCT   : https://www.ocbase.com/download
             edizione "Personal", Windows -> Programmi\OCCT\OCCT.exe

  Dopo la preparazione NON serve più Internet.


------------------------------------------------------------------------------
3. SUL POSTO: COSA DIRE AL VENDITORE
------------------------------------------------------------------------------

  "Ci metto 10-15 minuti. Uso programmi portatili dalla chiavetta: non
  installo niente e non tocco i tuoi file. Faccio girare un test grafico
  di circa 8 minuti, come una partita a un videogioco."

  Chiedi che il portatile:
    - sia collegato al SUO ALIMENTATORE ORIGINALE ASUS (spinotto tondo,
      180-200 W indicati sull'etichetta). La ricarica USB-C non basta per
      il test: la GPU andrebbe a potenza ridotta;
    - abbia Windows avviato e sbloccato;
    - possibilmente sia stato acceso da poco o a riposo (non appena usato
      per giocare).


------------------------------------------------------------------------------
4. PROCEDURA CRONOMETRATA (massimo 15 minuti)
------------------------------------------------------------------------------

  [00:00 - 01:00]  PREPARAZIONE
    - Alimentatore originale collegato (spia di ricarica accesa).
    - Chiudi giochi e programmi pesanti aperti.
    - Premi Fn+F5 fino a selezionare la modalità "Prestazioni"
      (Performance). "Turbo" va bene uguale; evita "Silenziosa".
    - Se Armoury Crate è aperto: Modalità GPU = "Standard" (NON "Eco",
      che spegne la RTX 4060).
    - Inserisci la chiavetta e apri la cartella KIT_TEST_LAPTOP.

  [01:00 - 02:30]  CONTROLLO AUTOMATICO
    - Doppio clic su 1_CONTROLLO_AUTOMATICO.bat -> alla richiesta di Windows
      rispondi "Sì" (serve per leggere la salute dell'SSD).
    - In circa un minuto controlla: modello, CPU, RAM, GPU e VRAM, SSD
      (salute, usura, ore di accensione), batteria, Gestione dispositivi,
      schermate blu, crash del driver video, errori hardware.
    - Verde = OK, Giallo = da valutare, Rosso = problema.
    - Il rapporto finisce in Rapporti\Controllo_....txt

  [02:30 - 04:00]  GPU-Z (identità della scheda video)
    - Apri Programmi\GPU-Z\GPU-Z.exe -> alla domanda di installazione
      rispondi "No".
    - In alto a destra, nel menu a tendina in basso, scegli
      "NVIDIA GeForce RTX 4060 Laptop GPU" (non la Radeon integrata).
    - Confronta con la tabella del capitolo 5. Poi chiudi GPU-Z.

  [04:00 - 05:00]  HWiNFO (sensori) + avvio registrazione
    - Apri Programmi\HWiNFO64\HWiNFO64.exe (rispondi "Sì" ai permessi).
      Se compare una finestrella iniziale: spunta "Sensors-only" e "Start".
    - Annota i valori A RIPOSO: temperatura CPU e GPU (vedi capitolo 7).
    - Premi il pulsante di LOGGING in basso (icona con il foglio e il "+",
      "Start Logging") e salva il file nella cartella Rapporti della
      chiavetta, es. test_FA507NV.csv.
    - Premi il pulsante con l'orologio ("Reset") per azzerare min/max.

  [05:00 - 06:00]  OCCT (test GPU)
    - Apri Programmi\OCCT\OCCT.exe (rispondi "Sì" ai permessi). Al primo
      avvio conferma l'uso personale/gratuito: NON serve alcun account.
    - Scegli il test "3D Standard" (oppure "3D Adaptive").
    - Dispositivo/GPU: "NVIDIA GeForce RTX 4060 Laptop GPU".
    - Se c'è l'opzione di rilevamento errori ("Error detection"),
      attivala. Lascia le altre impostazioni predefinite.
    - Durata: 8-10 minuti (o illimitata, la fermi tu).
    - Premi START (play).
    - Subito dopo: doppio clic su 2_MONITOR_GPU_8_MINUTI.bat
      (registra la GPU ogni 2 secondi e alla fine dà un giudizio).

  [06:00 - 14:00]  OSSERVAZIONE (il test lavora da solo)
    Guarda lo schermo e HWiNFO:
    - ARTEFATTI: puntini colorati, triangoli "sparati", righe, sfarfallii,
      schermo nero. NON devono comparire.
    - OCCT: il contatore errori deve restare a 0, il test non deve fermarsi.
    - HWiNFO: temperature e frequenze (capitolo 6 e 7).
    - Nella finestra del monitor GPU: le righe GIALLE/ROSSE indicano
      surriscaldamento o rallentamenti hardware.
    Intanto, senza interrompere il test (minuto 8-12):
    - Le ventole devono salire di giri entro 30-60 secondi (rumore
      aumentato è NORMALE). Ronzii metallici o ticchettii = ventola usurata.
    - Dalle griglie posteriori/laterali deve uscire aria calda.
    - Nessun odore di bruciato. La zona sopra la tastiera è calda, ma
      il poggiapolsi deve restare tiepido.
    - Guarda cerniere, scocca, angoli, viti mancanti, porte.

  [14:00 - 15:00]  FINE E DECISIONE
    - Ferma OCCT (se non si è fermato da solo) e guarda il riepilogo.
    - Il monitor GPU mostra il giudizio (verde/giallo/rosso).
    - In HWiNFO ferma il logging (stesso pulsante) e chiudi HWiNFO.
    - Doppio clic su 3_ANALIZZA_LOG_HWINFO.bat (pochi secondi).
    - Compila la CHECKLIST del capitolo 8.

  Facoltativo, solo se avanzano 2 minuti: in OCCT test "VRAM" per 2 minuti
  (controlla la memoria video; deve finire con 0 errori).

  NOTA: questo NON è uno stress test estremo. "3D Standard" carica la GPU
  come un gioco impegnativo per pochi minuti: abbastanza per far emergere
  GPU difettose, VRAM instabili, ventole o pasta termica da rifare, ma
  senza "torturare" il portatile del venditore.


------------------------------------------------------------------------------
5. GPU-Z: VALORI ATTESI (RTX 4060 Laptop GPU)
------------------------------------------------------------------------------

  Voce in GPU-Z        Valore atteso                    Se diverso...
  -------------------  -------------------------------  ---------------------
  Name                 NVIDIA GeForce RTX 4060          "RTX 4050" = modello
                       Laptop GPU                       diverso (FA507NU)
  GPU                  AD107                            altro chip = NO
  Technology           5 nm                             -
  Shaders              3072 Unified                     2560 = RTX 4050
  ROPs/TMUs            48 / 96                          -
  Memory Type          GDDR6 (Samsung/Hynix/Micron)     -
  Memory Size          8192 MB                          6144 MB = RTX 4050
  Bus Width            128 bit                          96 bit = RTX 4050
  Bus Interface        PCIe x8 4.0 @ x8 1.1/4.0         "@ x8 1.1" a riposo è
                                                        NORMALE (risparmio)
  Boost (clock)        circa 2300-2400 MHz              -
  Subvendor            ASUS                             altro = vBIOS strano
  Driver Version       NVIDIA, possibilmente recente    driver mancante = NO
  NVIDIA DLSS          presente                         -

  - A destra del nome NON deve esserci il simbolo di avviso giallo
    (GPU-Z lo mostra per schede contraffatte/non riconosciute).
  - Scheda "Sensors" > "PerfCap Reason" sotto carico:
      Pwr      = limite di potenza -> NORMALE
      VRel/VOp = limite di tensione -> NORMALE
      Thrm     = rallenta per calore -> se fisso per minuti, problema
                 di raffreddamento


------------------------------------------------------------------------------
6. HWiNFO: VALORI DA OSSERVARE
------------------------------------------------------------------------------

  Colonne: Current (attuale), Minimum, Maximum, Average (media).
  Guarda soprattutto MAXIMUM e AVERAGE dopo 5-8 minuti di test.

  Gruppo "CPU [#0]: AMD Ryzen 7 7735HS: Enhanced"
    CPU (Tctl/Tdie)               temperatura CPU (vedi capitolo 7)
    CPU Package Power / CPU PPT   consumo CPU: riposo 5-15 W, carico 35-80 W
    Core Clocks / Effective Clock frequenza: sotto carico 3,5-4,7 GHz
    Thermal Throttling (HTC)      "Yes" brevi = ok; "Yes" fisso = calore
    Thermal Throttling (PROCHOT)  deve essere "No"

  Gruppo "GPU [#0]: NVIDIA GeForce RTX 4060 Laptop"
    GPU Temperature               temperatura GPU (vedi capitolo 7)
    GPU Hot Spot Temperature      se presente: di solito 10-15 °C sopra
    GPU Power                     sotto carico: 80-115 W in Prestazioni,
                                  fino a ~140 W in Turbo (Dynamic Boost)
    GPU Clock                     sotto carico: 1800-2400 MHz, stabile
    GPU Memory Clock              2000 MHz (= 16 Gbps effettivi)
    GPU Core Load                 95-100% durante OCCT
    Performance Limit - Power     "Yes" = NORMALE sotto carico
    Performance Limit - Thermal   "Yes" per molto tempo = raffreddamento
                                  insufficiente
    Performance Limit - Reliability Voltage / Max Operating Voltage
                                  "Yes" = normale

  Ventole (gruppo "ASUS" / "ACPI" / "EC", se presente)
    CPU Fan / GPU Fan [RPM]       sotto carico circa 3500-6000 RPM.
                                  0 RPM a riposo può essere normale.

  Gruppo "S.M.A.R.T.: ... (SSD)" / "Drive"
    Drive Temperature             riposo 30-50 °C, sotto 70 °C
    Drive Remaining Life          meglio sopra 90%
    Drive Failure / Warning       devono essere "No"

  Gruppo "Windows Hardware Errors (WHEA)"
    Total Errors                  deve essere 0

  Gruppo "Battery"
    Wear Level                    usura batteria: sotto 20% ottimo,
                                  20-40% normale per l'età, oltre 40% da
                                  sostituire (argomento per trattare)

  SEGNALI DI ALLARME TIPICI
    - GPU Clock che crolla (es. da 2200 a <1500 MHz) mentre la temperatura
      GPU è a 87 °C o più  -> dissipatore intasato o pasta termica secca.
    - GPU Power bloccata a 50-60 W in modalità Prestazioni con l'alimentatore
      originale -> problema di alimentazione/limite (indaga).
    - CPU sempre a 95 °C con frequenza sotto i 3 GHz -> raffreddamento scarso.
    - Temperature che salgono senza fermarsi minuto dopo minuto (non si
      stabilizzano) -> ventole/dissipatore.


------------------------------------------------------------------------------
7. TEMPERATURE: NORMALI O PREOCCUPANTI? (TUF A15 FA507NV)
------------------------------------------------------------------------------

  Questo portatile è sottile per la sua potenza: scalda e fa rumore, è
  NORMALE. Il Ryzen 7 7735HS è progettato per lavorare fino a 95 °C, e la
  RTX 4060 Laptop ha un limite termico di circa 87 °C: arrivare vicino a
  questi valori sotto carico non è un guasto, il problema è quando li
  supera o quando, per restarci sotto, taglia molto le prestazioni.

  CPU (CPU Tctl/Tdie)
    A riposo (desktop)          40-60 °C   normale (sbalzi rapidi normali)
                                > 70 °C    preoccupante a riposo
    Durante il test GPU         60-85 °C   normale
                                85-95 °C   accettabile (modalità Turbo)
                                95 °C fisso + frequenza < 3 GHz -> problema

  GPU (GPU Temperature) dopo 5-8 minuti di OCCT 3D
    A riposo                    35-55 °C   normale (oppure sensore spento:
                                           la GPU dorme, è normale)
    Sotto carico                65-80 °C   OTTIMO
                                81-87 °C   NORMALE (fascia alta)
                                88-91 °C   ATTENZIONE: al limite, controlla
                                           se la frequenza crolla
                                >= 92 °C   PROBLEMA: raffreddamento
                                           inefficiente
    GPU Hot Spot (se presente)  fino a 95-100 °C ok, oltre 105 °C problema

  SSD (Drive Temperature)
    < 50 °C riposo, < 70 °C sotto carico. Oltre 75-80 °C rallenta.

  Superficie
    Tastiera centrale/alta calda (40-48 °C) normale; poggiapolsi tiepido.

  Confronto utile: una GPU che a 8 minuti è a 75-82 °C con frequenza
  stabile intorno a 2000+ MHz e consumo 90-115 W è in ottima salute.


------------------------------------------------------------------------------
8. CHECKLIST FINALE: COMPRARE O NO?
------------------------------------------------------------------------------

  A) BLOCCANTI - se anche UNO è vero, NON comprare (o solo a prezzo molto
     più basso, sapendo cosa fai):
  [ ] Modello non FA507NV/FA507NVR, oppure GPU RTX 4050 / VRAM 6 GB
  [ ] GPU NVIDIA assente in Gestione dispositivi, o "Codice 43"
      (prima verifica che in Armoury Crate la Modalità GPU non sia "Eco")
  [ ] Artefatti a schermo, schermo nero, crash o riavvio durante OCCT
  [ ] OCCT segnala ERRORI (anche uno solo)
  [ ] GPU >= 92 °C, oppure frequenza GPU che crolla sotto ~1500 MHz
  [ ] Ventola che non gira / rumore metallico forte / odore di bruciato
  [ ] SSD "non integro", errori non corretti o usura >= 80%
  [ ] Numero di serie del BIOS diverso dall'etichetta sotto il portatile
  [ ] Molti crash del driver video o schermate blu recenti e il venditore
      non sa spiegarli

  B) DA VALUTARE / ARGOMENTI PER TRATTARE IL PREZZO:
  [ ] Batteria sotto 80% di salute (sostituzione circa 60-100 euro)
  [ ] Un solo modulo RAM (single channel: +1 modulo 8 GB circa 25-40 euro)
  [ ] GPU 88-91 °C o CPU sempre a 95 °C -> pulizia e pasta termica
      (circa 40-80 euro in laboratorio)
  [ ] Alimentatore non originale o mancante (originale circa 40-60 euro)
  [ ] Pixel morti, aloni sullo schermo, tasti che non funzionano
  [ ] Cerniere lente, crepe, viti mancanti, scocca segnata
  [ ] SSD con usura 40-79% o moltissime ore di accensione
  [ ] Windows non attivato

  C) TUTTO OK SE:
  [ ] Controllo automatico senza righe rosse
  [ ] GPU-Z: RTX 4060 Laptop, AD107, 3072 shader, 8192 MB GDDR6, 128 bit
  [ ] OCCT 8 minuti: 0 errori, nessun artefatto, nessun crash
  [ ] GPU sotto 88 °C con frequenza stabile e consumo 80+ W
  [ ] CPU sotto 95 °C (o ai 95 °C solo a tratti)
  [ ] Ventole che salgono di giri e aria calda in uscita
  [ ] Monitor GPU e analisi HWiNFO senza righe rosse

  Se C è tutto spuntato e A è vuoto: il portatile ha SUPERATO il controllo.
  Ricorda: questo test esclude problemi EVIDENTI di GPU, raffreddamento e
  stabilità in 15 minuti; non può garantire l'assenza di guasti futuri.

  Dopo l'acquisto: fatti dare scatola/scontrino se ci sono, poi fai un
  ripristino completo di Windows (Impostazioni > Sistema > Ripristino)
  e installa Armoury Crate/MyASUS e l'ultimo driver NVIDIA.


------------------------------------------------------------------------------
9. CONTROLLI EXTRA (se il venditore concede 3-5 minuti in più)
------------------------------------------------------------------------------

  - Schermo: apri Extra\test_schermo.html (si apre con Edge, offline).
    Premi F11 per lo schermo intero, clicca o usa le frecce per cambiare
    colore. Cerca pixel morti (puntini fissi), aloni chiari agli angoli
    sul nero (un po' di "IPS glow" è normale), macchie.
  - Tastiera: apri Extra\test_tastiera.html e premi tutti i tasti: ognuno
    diventa verde quando funziona. Prova anche il touchpad (clic e scorrimento).
  - Porte USB: inserisci la chiavetta in ogni porta e verifica che appaia.
  - Webcam: app "Fotocamera". Audio: un video o un suono qualsiasi.
  - Wi-Fi: attiva l'hotspot del telefono e collegati.
  - Etichetta alimentatore: ASUS, potenza 180-200 W, spinotto integro.
  - Schermo: nelle Impostazioni > Schermo > Avanzate deve risultare 144 Hz (165 Hz su
    alcune versioni). 60 Hz = schermo impostato male o non originale.


------------------------------------------------------------------------------
10. PROBLEMI COMUNI E SOLUZIONI
------------------------------------------------------------------------------

  "Windows ha protetto il PC" quando apro un .bat
    -> "Ulteriori informazioni" > "Esegui comunque". (A casa, l'opzione
       "Annulla blocco" sullo ZIP evita il problema.)

  Il .bat si apre e si chiude subito / lo script è bloccato
    -> Usa i programmi a mano (GPU-Z, HWiNFO, OCCT): la procedura funziona
       lo stesso, leggendo i valori dei capitoli 5-7.

  La RTX 4060 non compare (né in GPU-Z, né in OCCT)
    -> Armoury Crate > Modalità GPU: "Standard" (non "Eco"). Se manca il
       driver NVIDIA ("Scheda video di base Microsoft") il test GPU non è
       possibile: chiedi al venditore di installare il driver o valuta di
       rinunciare.

  OCCT usa la grafica integrata (Radeon) invece della RTX
    -> Seleziona la GPU NVIDIA nel menu del test. Se non basta:
       Impostazioni Windows > Schermo > Grafica > aggiungi OCCT.exe >
       "Prestazioni elevate".

  Il monitor GPU dice "GPU non sotto carico"
    -> OCCT non era avviato sulla RTX 4060, o è stato avviato dopo.
       Riavvia il monitor con il test in corso.

  Consumo GPU basso (sotto 60 W)
    -> Controlla: alimentatore originale collegato? Modalità "Prestazioni"
       o "Turbo" (Fn+F5)? Batteria non in modalità risparmio?

  HWiNFO non mostra le ventole
    -> Su alcuni ASUS i giri ventola non sono leggibili: verifica a orecchio
       e sentendo l'aria calda in uscita.


------------------------------------------------------------------------------
11. VERSIONI DEI PROGRAMMI E FONTI UFFICIALI
------------------------------------------------------------------------------

  Lo script di preparazione scarica sempre l'ULTIMA versione disponibile
  sui siti ufficiali e registra versione, firma digitale e SHA256 in
  Rapporti\VERSIONI_PROGRAMMI.txt.

  Versioni attuali rilevate a novembre 2026 (indicative):
    GPU-Z            2.7x (2.71.0, settembre 2026)
    HWiNFO64         8.5x (8.50 o successiva)
    OCCT Personal    17.x (17.0.18)

  Siti ufficiali (gli UNICI da usare):
    https://www.techpowerup.com/download/techpowerup-gpu-z/
    https://www.hwinfo.com/download/
    https://www.ocbase.com/download

  Firme digitali attese (le controlla lo script):
    GPU-Z.exe     -> TechPowerUp
    HWiNFO64.exe  -> Martin Malík - REALiX
    OCCT.exe      -> OCBASE

  Non scaricare questi programmi da siti di download generici: spesso
  aggiungono installer con pubblicità o software indesiderato.
==============================================================================
