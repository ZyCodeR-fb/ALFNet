<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">模型</th>
      <th colspan="3" scope="colgroup">METR-LA</th>
      <th colspan="3" scope="colgroup">PEMS-BAY</th>
      <th colspan="3" scope="colgroup">PEMS04</th>
      <th colspan="3" scope="colgroup">PEMS08</th>
    </tr>
    <tr>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ALFNet</td>
      <td><b>3.842</b></td><td><b>6.085</b></td><td><b>10.337</b></td>
      <td><b>2.723</b></td><td><b>4.201</b></td><td><b>6.276</b></td>
      <td>39.565</td><td>53.346</td><td><b>23.257</b></td>
      <td>38.749</td><td>48.976</td><td><b>43.318</b></td>
    </tr>
    <tr>
      <td>w/o State Age</td>
      <td>3.878</td><td><u>6.097</u></td><td>10.423</td>
      <td>2.732</td><td>4.248</td><td>6.464</td>
      <td><u>39.314</u></td><td><u>52.436</u></td><td>23.955</td>
      <td>39.770</td><td>50.360</td><td>54.941</td>
    </tr>
    <tr>
      <td>w/o Adaptive Routing</td>
      <td>3.920</td><td>6.127</td><td>10.602</td>
      <td>2.770</td><td>4.280</td><td>6.514</td>
      <td>39.826</td><td>53.401</td><td><u>23.415</u></td>
      <td><u>37.869</u></td><td><u>48.160</u></td><td><u>45.604</u></td>
    </tr>
    <tr>
      <td>w/o Spatial Context</td>
      <td>4.400</td><td>6.645</td><td>12.529</td>
      <td>2.958</td><td>4.493</td><td>7.011</td>
      <td>40.292</td><td>54.694</td><td>24.877</td>
      <td>40.366</td><td>51.031</td><td>50.564</td>
    </tr>
    <tr>
      <td>w/o ALFTemporal–FiLM</td>
      <td>4.264</td><td>6.560</td><td>11.760</td>
      <td>2.890</td><td>4.487</td><td>6.992</td>
      <td>42.137</td><td>56.293</td><td>26.361</td>
      <td>41.122</td><td>52.684</td><td>53.160</td>
    </tr>
    <tr>
      <td>w/o Graph Refinement</td>
      <td><u>3.852</u></td><td>6.106</td><td><u>10.385</u></td>
      <td><u>2.726</u></td><td>4.258</td><td>6.525</td>
      <td>40.032</td><td>53.743</td><td>23.509</td>
      <td>37.885</td><td><b>47.909</b></td><td>46.145</td>
    </tr>
    <tr>
      <td>Pointwise Refinement</td>
      <td>3.911</td><td>6.130</td><td>10.765</td>
      <td>2.745</td><td>4.258</td><td>6.495</td>
      <td>39.758</td><td>53.440</td><td>23.441</td>
      <td><b>37.268</b></td><td>48.161</td><td>47.338</td>
    </tr>
    <tr>
      <td>Stop Temporal→Recon Gradient</td>
      <td>3.924</td><td>6.151</td><td>10.504</td>
      <td>2.752</td><td><u>4.233</u></td><td><u>6.299</u></td>
      <td><b>39.216</b></td><td><b>52.389</b></td><td>23.851</td>
      <td>39.712</td><td>51.585</td><td>49.805</td>
    </tr>
  </tbody>
</table>
