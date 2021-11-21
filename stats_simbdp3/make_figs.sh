#!/bin/sh
VERSION="0.0.2" # Time-stamp: <2021-11-21T18:03:50Z>

DATE=`date +%Y%m%d`

set `seq -w 1 99`

set -x
set -e

python plot_logs.py normal fana -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana -p AccAbortion -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana -p AccTemple -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana fanampr fanamprmprs -p Education -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana fanampr fanamprmprs -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana fanampr fanamprmprs -p Population -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana fanampr fanamprmprs -p AccAbortion -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal fana fanampr fanamprmprs -p Power -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p AccDeathRate -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p AccAbortion -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p AccBreakup -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p Welfare -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p Budget -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p Power -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarmpopmamb mwarmpop -p Injured -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarfpr -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarfpr -p AccAbortion -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarfpr -p Education -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal mwar mwarfpr -p Power -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal ledu ltom ledultom -p AccKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal ledu ltom ledultom -p NewKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal ledu ltom ledultom -p Education -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lpe -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py lpe ltomlpe ledultomlpe -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal ledu ltomlpe ledultomlpe -p AccKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p Population -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p AccKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p NewKarma2 -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p Hating -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p VirtualHating -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal lsth ltom ltomlsth -p AccVKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal worst1 worst2 -p AccDeath -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal worst1 worst2 -p AccAbortion -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal worst1 worst2 -p AccKarma -o fig-${DATE}_$1.png
shift 1
python plot_logs.py normal worst1 worst2 -p Education -o fig-${DATE}_$1.png
shift 1
