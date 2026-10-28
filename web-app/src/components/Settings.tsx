import React from "react";
import { FiCheck, FiArrowUpRight } from "react-icons/fi";
import { schemes, themes, useAppearance } from "../appearance";

export default function Settings() {
  const { theme, scheme, loading, message, change } = useAppearance();
  return <div className="settings-page">
    <div className="settings-heading"><h1>Nastavitve videza</h1><p>Izberi postavitev in barve za svoj pregled novic.</p></div>
    <fieldset className="appearance-section" disabled={loading}>
      <legend>Tema</legend><p className="section-description">Spremeni postavitev, pisave in značaj celotne strani.</p>
      <div className="theme-options">{themes.map(item => <label className={`theme-option preview-${item.id}`} key={item.id}>
        <input type="radio" name="appearance-theme" aria-label={item.name} value={item.id} checked={theme === item.id} onChange={() => change({ theme: item.id, scheme })} />
        <span className="theme-preview" aria-hidden="true"><span className="preview-nav"><b>novice/</b><i /><i /><i /></span><span className="preview-filter"><i /><i /><i /><i /></span><span className="preview-content"><span /><span /><span /></span></span>
        <span className="theme-option-heading"><strong>{item.name}</strong><span className="theme-selected">{theme === item.id ? <><FiCheck aria-hidden="true" />Izbrano</> : "Izberi"}</span></span>
        <span className="theme-detail">{item.detail}</span><span className="theme-description">{item.description}</span>
      </label>)}</div>
    </fieldset>
    <fieldset className="appearance-section" disabled={loading}>
      <legend>Barvna shema</legend><p className="section-description">Vsak dizajn ima svojo svetlo in temno paleto. Izvirna ohrani njegov začetni videz.</p>
      <div className="scheme-options">{schemes.map(item => <label className={`scheme-option swatch-${item.id}`} key={item.id}>
        <input type="radio" name="appearance-scheme" aria-label={item.name} value={item.id} checked={scheme === item.id} onChange={() => change({ theme, scheme: item.id })} />
        <span className="scheme-swatches" aria-hidden="true"><i /><i /><i /></span><span><strong>{item.name}</strong><small>{item.description}</small></span>{scheme === item.id && <FiCheck className="scheme-check" aria-hidden="true" />}
      </label>)}</div>
    </fieldset>
    <p className="appearance-message" role="status">{message || "Izbira se samodejno shrani v tem brskalniku."}</p>
    <section className="about-project" aria-labelledby="about-title"><div><h2 id="about-title">O projektu</h2><p>Agregator in arhiv slovenskih spletnih novic. Prebiraj različne vire, poišči novice na zemljevidu ali raziskuj podatke z grafi.</p></div><a className="button quiet" href="https://github.com/sol-projects/novice" target="_blank" rel="noopener noreferrer">Izvorna koda<FiArrowUpRight aria-hidden="true" /></a></section>
  </div>;
}
