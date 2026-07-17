#!/bin/bash
sleep 2
JACK_STATE=$(amixer -c 0 cget iface=CARD,name='Headphone Jack' | grep -o 'values=on\|values=off')
if [ "$JACK_STATE" = "values=on" ]; then
    amixer -c 0 cset iface=MIXER,name='Headphone Switch' on
else
    amixer -c 0 cset iface=MIXER,name='Headphone Switch' off
fi

# Force the DPCM front-end/back-end link to establish
speaker-test -t wav -c 2 -D plughw:0,0 -l 1h