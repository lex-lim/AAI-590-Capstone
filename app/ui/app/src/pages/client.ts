import Anthropic from '@anthropic-ai/sdk'
import {apikey} from '../../../../../key'


export const client = new Anthropic({
  apiKey: apikey,
  dangerouslyAllowBrowser: true
})