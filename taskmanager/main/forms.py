from django import forms
from django.core.exceptions import ValidationError

class TickerForm(forms.Form):
    tickers = forms.CharField(label='Enter a ticker', required=False)
    start_date = forms.DateField(label='Start Date', widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(label='End Date', widget=forms.DateInput(attrs={'type': 'date'}))

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")

        if start_date and end_date:
            if start_date > end_date:
                raise ValidationError("Start date must be before end date.")

        return cleaned_data

class IndicatorForm(forms.Form):
    indicators = forms.ChoiceField(choices=[('sma', 'SMA'), ('ema', 'EMA'), ('rsi', 'RSI'), ('bb', 'BB'), ('macd', 'MACD')], widget=forms.Select(attrs={'class': 'form-control'}), required=False)
class TickersVol(forms.Form):
    tickers = forms.CharField(label='Enter up to 10 tickers', max_length=200, required=False)

class TickersVolume(forms.Form):
    tickers = forms.CharField(label='Enter up to 10 tickers', max_length=200, required=False)